import pandas
import os
import os.path
import re
import io
import torch

from .utils import fenToTensor

line_parse_re = re.compile(r"tensor\(([0-9.]+), .+?\)")

default_vals_mean = {
    'cp_rel' : 0.7,
    'clock_percent' : 0.35,
    'move_ply' : 30,
    'opponent_elo' : 1500,
    'active_elo' : 1500,
}

def parse_tensor(t):
    return t.group(1)

line_parse_re = re.compile(r"tensor\(([0-9.]+), .+?\)")

def parse_tensor(t):
    return t.group(1)

class ModelWrapper(object):
    def __init__(self, path, default_vals = None):
        self.path = path.rstrip('/')
        model_paths = [fname for fname in os.listdir(self.path) if fname.startswith('net')]
        self.model_paths = sorted(model_paths,
                                      key = lambda x : int(x.split('-')[-1].split('.')[0])
                                          if 'final' not in x else float('-inf'))
        self.newest_save = self.find_best_save()
        self.name = os.path.basename(self.path)
        self.net = torch.load(self.newest_save)
        if torch.cuda.is_available():
            self.net = self.net.cuda()
        else:
            self.net = self.net.cpu()

        self.has_extras = False
        self.extras = None
        self.default_vals = {}
        self.defaults_dict = None

        if default_vals is not None:
            self.default_vals = default_vals_mean.copy()
        try:
            if self.net.has_extras:
                self.has_extras = True
                self.extras = self.net.extra_inputs.copy()
                self.defaults_dict = {}
                for n in self.extras:
                    self.defaults_dict[n] = torch.Tensor([self.default_vals.get(n, default_vals_mean[n])])
        except AttributeError:
            pass

    def find_best_save(self):
        return os.path.join(self.path, self.model_paths[-1])

    def __repr__(self):
        return f"<ModelWrapper {self.name}>"

    def run_batch(self, input_fens,  new_defaults = None):
        input_tensors = torch.stack([fenToTensor(fen) for fen in input_fens])
        extra_x = None
        if self.has_extras:
            extra_x = {}
            if new_defaults is None:
                for n in self.extras:
                    extra_x[n] = self.defaults_dict[n] * torch.ones([input_tensors.shape[0]], dtype = torch.float32)
            else:
                for n in self.extras:
                    extra_x[n] = new_defaults.get(n, self.defaults_dict[n]) * torch.ones([input_tensors.shape[0]], dtype = torch.float32)
        if torch.cuda.is_available():
            input_tensors = input_tensors.cuda()
            if extra_x is not None:
                for n in self.extras:
                    extra_x[n] = extra_x[n].cuda()
        ret_dat = self.net.dict_forward(input_tensors, extra_x = extra_x)

        for n in ['is_blunder_wr', 'is_blunder_mean']:
            try:
                return ret_dat[n].detach().cpu().numpy()
            except KeyError:
                pass
        for n in ['winrate_loss', 'is_blunder_wr_mean']:
            try:
                return ret_dat[n].detach().cpu().numpy()  * 5
            except KeyError:
                pass
        raise KeyError(f"No known output types found in: {ret_dat.keys()}")
