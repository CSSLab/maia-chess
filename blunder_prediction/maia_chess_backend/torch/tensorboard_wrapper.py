import torch
#import torch.utils.tensorboard
import tensorboardX
import os.path

import pytz
tz = pytz.timezone('Canada/Eastern')

class TB_wrapper(object):
    """Defers creating files until used"""

    def __init__(self, name, log_dir = 'runs'):
        self.log_dir = log_dir
        self.name = name
        self._tb = None

    @property
    def tb(self):
        if self._tb is None:
            tb_path = os.path.join(self.log_dir, f"{self.name}")
            if os.path.isdir(tb_path):
                i = 2
                tb_path = tb_path + f"_{i}"
                while os.path.isdir(tb_path):
                    i += 1
                    #only works to 10
                    tb_path = tb_path[:-2] + f"_{i}"
            self._tb = tensorboardX.SummaryWriter(
                log_dir = tb_path
                )
            #_{datetime.datetime.now(tz).strftime('%Y-%m-%d-H-%M')}"))
        return self._tb

    def add_scalar(self, *args, **kwargs):
        return self.tb.add_scalar(*args, **kwargs)

    def add_graph(self, model, input_to_model):
        self.tb.add_graph(model, input_to_model = input_to_model, verbose = False)

    def add_histogram(self, *args, **kwargs):
        return self.tb.add_histogram(*args, **kwargs)

    def flush(self):
        self.tb.flush()
