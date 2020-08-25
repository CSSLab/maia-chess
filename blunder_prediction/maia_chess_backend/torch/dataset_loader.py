import torch
import torch.utils.data

import os
import os.path
import glob
import json

import numpy as np
import pandas

from ..utils import profile_helper

def load_mmap_np(mmap_name):
    mmap_name = os.path.abspath(mmap_name)
    mmap_bname = os.path.basename(mmap_name)
    s_split = mmap_bname.split('.')[0].split('+')
    if len(s_split) == 3:
        mmapType, np_type, a_len = s_split
        return np.memmap(mmap_name,
               dtype = np_type,
               mode = 'r',
               shape = (int(a_len),) )
    mmapType, a_len = s_split
    if mmapType == 'board':
        return np.memmap(mmap_name,
                   dtype = np.bool,
                   mode = 'r',
                   shape = (int(a_len), 17, 8, 8))
    elif mmapType in ['move', 'top_nonblunder', 'top_blunder']:
        return np.memmap(mmap_name,
                   dtype = np.int,
                   mode = 'r',
                   shape = (int(a_len),))
    else:
        raise RuntimeError(f'Invalid mmap path name: {mmap_name}')

class MmapSingleLoader(object):
    @profile_helper
    def __init__(self, target_name, y_names, batch_size, max_rows = None,  max_samples = None, linear_mode = False):
        self.target_name = target_name
        self.max_samples = max_samples
        self.max_rows = max_rows
        self.linear_mode = linear_mode
        self.y_names = y_names.copy()
        if 'game_id' in self.y_names:
            self.y_names.remove('game_id')
            self.with_game_id = True
            with open(self.make_rel_path('game_id_lookup.json')) as f:
                self.ids_map = json.load(f)
        else:
            self.with_game_id = False
        self.batch_size = batch_size

        self.board_array = load_mmap_np(self.make_rel_path("board+*.mm"))

        self.y_vals = {}
        try:
            for name in self.y_names:
                self.y_vals[name] = load_mmap_np(self.make_rel_path(f"{name}+*.mm"))
            if self.with_game_id:
                self.y_vals['game_id'] = load_mmap_np(self.make_rel_path("game_id+*.mm"))
        except IndexError:
            raise FileNotFoundError(f"{target_name} is missing {name} value")
        if max_rows is not None:
            self.board_array = self.board_array[:max_rows]
            for name in list(self.y_vals.keys()):
                self.y_vals[name] = self.y_vals[name][:max_rows]

        self.num_blocks = self.board_array.shape[0] // self.batch_size
        if self.max_samples:
            self.num_blocks = min(self.max_samples, self.num_blocks)
        self.batch_lst = list(range(self.num_blocks))
        np.random.shuffle(self.batch_lst)
        if self.linear_mode:
            self.batch_lst = sorted(self.batch_lst)

    def make_rel_path(self, name):
        return glob.glob(os.path.join(self.target_name, name))[0]

    def load_testing(self):
        return self.get_index(0)

    @profile_helper
    def get_index(self, index):
        ret_board = self.board_array[self.batch_size * index: self.batch_size * (index + 1)]
        #ret_board = torch.from_numpy(ret_board)
        #ret_board = ret_board.pin_memory()
        ret_ys = {}
        for name in self.y_names:
            a_mm = self.y_vals[name][self.batch_size * index: self.batch_size * (index + 1)]
            ret_ys[name] = a_mm#torch.from_numpy(a_mm)
        if self.with_game_id:
            a_mm = self.y_vals['game_id'][self.batch_size * index: self.batch_size * (index + 1)]
            ret_ids = [self.ids_map[str(i)] for i in a_mm]
            ret_ys['game_id'] = ret_ids
        return ret_board, ret_ys

    @profile_helper
    def __next__(self):
        try:
            batch_num = self.batch_lst.pop()
        except IndexError:
            raise StopIteration(f"out of batches in {self.target_name}")
        return self.get_index(batch_num)

class MmapIterLoaderMap(torch.utils.data.IterableDataset):
    def __init__(self, target_dir, y_names, mini_batch_size, max_rows = None, max_samples = None, linear_mode = False):
        super().__init__()
        self.target_dir = target_dir
        self.target_names = os.listdir(target_dir)
        if y_names == 'all':
            vals = os.listdir(os.path.join(target_dir, self.target_names[0], 'blunder'))
            y_names = []
            for v in (n for n in vals if n.endswith('mm')):
                name = v.split('+')[0]
                if name not in ['game_id', 'board']:
                    y_names.append(name)
        self.y_names = y_names.copy()
        self.max_rows = max_rows
        self.max_samples = max_samples
        self.mini_batch_size = mini_batch_size
        self.linear_mode = linear_mode

        #add more .copy() to taste

        self.blunder_names = sorted(self.target_names).copy()
        self.current_blunders = self.blunder_names.copy()
        np.random.shuffle(self.current_blunders)

        self.nonblunder_names = sorted(self.target_names).copy()
        self.current_nonblunders = self.nonblunder_names.copy()
        np.random.shuffle(self.current_nonblunders)

        if self.linear_mode:
            self.current_blunders = sorted(self.current_blunders)
            self.current_nonblunders = sorted(self.current_nonblunders)

        self.blunder_loader = None
        self.nonblunder_loader = None

        self.init_next_blunder_loader()
        self.init_next_nonblunder_loader()
        self.next_is_blunder = True

        self.num_blunders = 0
        for b_name in self.blunder_names:
            self.num_blunders += self.get_len(os.path.join(b_name, 'blunder'))

        self.num_nonblunders = 0
        for nb_name in self.nonblunder_names:
            self.num_nonblunders += self.get_len(os.path.join(nb_name, 'nonblunder'))

    def get_len(self, target):
        p = glob.glob(os.path.join(
                        self.target_dir,
                        target,
                        "board+*.mm"
                        ))[0]
        return int(p.split('+')[-1].split('.')[0])

    def __len__(self):
        return self.num_nonblunders + self.num_nonblunders

    def __repr__(self):
        return f"<MmapIterLoaderMap {len(self.blunder_names)} months {len(self)} samples>"

    @profile_helper
    def init_next_blunder_loader(self, depth = 0):
        try:
            self.blunder_loader = MmapSingleLoader(
                                        os.path.join(
                                                    self.target_dir,
                                                    self.current_blunders.pop(),
                                                    'blunder',
                                                    ),
                                        self.y_names,
                                        self.mini_batch_size // 2,
                                        max_rows = self.max_rows,
                                        max_samples = self.max_samples,
                                        linear_mode = self.linear_mode,
                                        )
        except IndexError:
            if depth > 10:
                raise FileNotFoundError("Blunder file not found")
            self.current_blunders = self.blunder_names.copy()
            np.random.shuffle(self.current_blunders)
            if self.linear_mode:
                self.current_blunders = sorted(self.current_blunders)
            self.init_next_blunder_loader(depth = depth + 1)

    @profile_helper
    def init_next_nonblunder_loader(self, depth = 0):
        try:
            self.nonblunder_loader = MmapSingleLoader(
                                        os.path.join(
                                                    self.target_dir,
                                                    self.current_nonblunders.pop(),
                                                    'nonblunder',
                                                    ),
                                        self.y_names,
                                        self.mini_batch_size // 2,
                                        max_rows = self.max_rows,
                                        max_samples = self.max_samples,
                                        linear_mode = self.linear_mode,
                                        )
        except IndexError:
            if depth > 10:
                raise FileNotFoundError("Blunder file not found")
            self.current_nonblunders = self.nonblunder_names.copy()
            np.random.shuffle(self.current_nonblunders)
            if self.linear_mode:
                self.current_nonblunders = sorted(self.current_nonblunders)
            self.init_next_nonblunder_loader(depth = depth + 1)

    @profile_helper
    def get_next_b(self):
        try:
            return next(self.blunder_loader)
        except StopIteration:
            self.init_next_blunder_loader()
            return self.get_next_b()

    @profile_helper
    def get_next_nb(self):
        try:
            return next(self.nonblunder_loader)
        except StopIteration:
            self.init_next_nonblunder_loader()
            return self.get_next_nb()

    @profile_helper
    def __iter__(self):
        while True:
            yield next(self)

    @profile_helper
    def __next__(self):
        x_blunder, y_blunder = self.get_next_b()
        x_nonblunder, y_nonblunder = self.get_next_nb()

        ret_x = np.stack([x_blunder, x_nonblunder], axis = 0)
        ret_x = torch.from_numpy(ret_x)
        ret_x = ret_x.cuda()
        ret_x = ret_x.reshape(-1, 17, 8, 8)
        ret_x = ret_x.float()
        ret_y = {}
        for k, y_b in y_blunder.items():
            k_a = np.stack([y_b, y_nonblunder[k]])
            ret_y[k] = torch.from_numpy(k_a).cuda().reshape(-1)

        return ret_x, ret_y

    def gen_big_batch_df(self, mini_batches_per, include_board = False):
        self.current_blunders = sorted(self.blunder_names.copy())
        self.current_nonblunders = sorted(self.blunder_names.copy())
        self.init_next_nonblunder_loader()
        self.init_next_blunder_loader()
        num_targets = len(self.blunder_names)

        vals = []

        for i in range(num_targets):
            for j in range(mini_batches_per):
                b, y = next(self)
                r = np.stack([y[k].cpu().numpy() for k in self.y_names], axis = 1)

                if include_board:
                    r = np.concatenate([r, b.cpu().numpy().reshape(len(b), -1)], axis = 1)
                vals.append(r)
            self.init_next_nonblunder_loader()
            self.init_next_blunder_loader()
        return pandas.DataFrame(np.concatenate(vals),
                 columns = self.y_names + ([f"board_{i:04.0f}" for i in range(r.shape[1] - len(self.y_names))] if include_board else [])
                )
