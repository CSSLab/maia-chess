import torch
import torch.utils.data

import os
import os.path
import glob
import json

import numpy as np

from ..utils import profile_helper

def load_mmap_np(mmap_name):
    mmap_name = os.path.abspath(mmap_name)
    s_split = mmap_name.split('.')[0].split('+')
    if len(s_split) == 4:
        name, mmapType, np_type, a_len = s_split
        return np.memmap(mmap_name,
               dtype = np_type,
               mode = 'r',
               shape = (int(a_len),) )
    name, mmapType, a_len = s_split
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

def get_len(target_path):
    p = glob.glob(f"{target_path}+board+*.mm")[0]
    return int(p.split('+')[-1].split('.')[0])

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
            with open(f'{target_name}_game_id_lookup.json') as f:
                self.ids_map = json.load(f)
        else:
            self.with_game_id = False
        self.batch_size = batch_size

        self.board_array = load_mmap_np(glob.glob(f"{target_name}+board+*.mm")[0])
        self.y_vals = {}
        try:
            for name in self.y_names:
                self.y_vals[name] = load_mmap_np(glob.glob(f"{target_name}+{name}+*.mm")[0])
            if self.with_game_id:
                self.y_vals['game_id'] = load_mmap_np(glob.glob(f"{target_name}+game_id+*.mm")[0])
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

class MmapIterLoaderMap_old(torch.utils.data.IterableDataset):
    def __init__(self, target_dir, y_names, mini_batch_size, max_rows = None, max_samples = None, linear_mode = False):
        super().__init__()
        self.target_dir = target_dir
        self.blunders_dir = os.path.join(self.target_dir, 'blunder')
        self.non_blunders_dir = os.path.join(self.target_dir, 'nonblunder')
        self.y_names = y_names
        self.max_rows = max_rows
        self.max_samples = max_samples
        self.mini_batch_size = mini_batch_size
        self.linear_mode = linear_mode

        self.blunder_names = set([n.split('+')[0] for n in glob.glob(os.path.join(self.blunders_dir,'*.mm'))])
        self.current_blunders = list(self.blunder_names)
        np.random.shuffle(self.current_blunders)

        self.nonblunder_names = set([n.split('+')[0] for n in glob.glob(os.path.join(self.non_blunders_dir,'*.mm'))])
        self.current_nonblunders = list(self.nonblunder_names)
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
            self.num_blunders += get_len(b_name)

        self.num_nonblunders = 0
        for nb_name in self.nonblunder_names:
            self.num_nonblunders += get_len(nb_name)

    def __len__(self):
        return self.num_nonblunders + self.num_nonblunders

    def __repr__(self):
        return f"<MmapIterLoaderMap {len(self.blunder_names)} months {len(self)} samples>"

    @profile_helper
    def init_next_blunder_loader(self):
        try:
            self.blunder_loader = MmapSingleLoader(
                                        self.current_blunders.pop(),
                                        self.y_names,
                                        self.mini_batch_size // 2,
                                        max_rows = self.max_rows,
                                        max_samples = self.max_samples,
                                        linear_mode = self.linear_mode,
                                        )
        except IndexError:
            self.current_blunders = list(self.blunder_names)
            np.random.shuffle(self.current_blunders)
            if self.linear_mode:
                self.current_blunders = sorted(self.current_blunders)
            self.init_next_blunder_loader()

    @profile_helper
    def init_next_nonblunder_loader(self):
        try:
            self.nonblunder_loader = MmapSingleLoader(
                                        self.current_nonblunders.pop(),
                                        self.y_names,
                                        self.mini_batch_size // 2,
                                        max_rows = self.max_rows,
                                        max_samples = self.max_samples,
                                        linear_mode = self.linear_mode,
                                        )
        except IndexError:
            self.current_nonblunders = list(self.nonblunder_names)
            np.random.shuffle(self.current_nonblunders)
            if self.linear_mode:
                self.current_nonblunders = sorted(self.current_nonblunders)
            self.init_next_nonblunder_loader()

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

    def get_xy_test(self):
        test_blunder = sorted(self.blunder_names)[0]
        blunder_loader = MmapSingleLoader(test_blunder,
                                           self.y_names,
                                           self.mini_batch_size,
                                           max_rows = self.max_rows,
                                           max_samples = self.max_samples,
                                          )
        test_nonblunder = sorted(self.nonblunder_names)[0]
        nonblunder_loader = MmapSingleLoader(test_nonblunder,
                                           self.y_names,
                                           self.mini_batch_size,
                                           max_rows = self.max_rows,
                                           max_samples = self.max_samples,
                                          )
        x_b, y_b = blunder_loader.load_testing()
        x_nb, y_nb = nonblunder_loader.load_testing()

        x_ret = torch.cat([x_b, x_nb], dim = 0)
        y_ret = {}
        for name in self.y_names:
            y_ret[name] = torch.cat([y_b[name], y_nb[name]], dim = 0)
        return x_ret, y_ret

class Raw_loader(object):
    def __init__(self, target_dir, y_names, mini_batch_size, max_rows = None):
        self.target_dir = target_dir
        self.y_names = y_names
        self.max_rows = max_rows
        self.files_names = set(glob.glob(f"target_dir/*.csv.bz2"))
        self.current_files = list(self.files_names)
        np.random.shuffle(self.current_files)

        self.loader = None
        self.init_loader()
        self.next_is_blunder = True

    def init_loader(self):
        try:
            self.loader = RawCSVLoader(self.current_files.pop(),
                                                   self.y_names,
                                                   self.mini_batch_size,
                                                   max_rows = self.max_rows,
                                                  )
        except IndexError:
            self.current_files = list(self.files_names)
            np.random.shuffle(self.current_files)
            self.init_loader()

    def get_next_b(self):
        try:
            return self.loader.get_next_b()
        except StopIteration:
            self.init_loader()
            return self.loader.get_next_b()

    def get_next_nb(self):
        try:
            return self.loader.get_next_nb()
        except StopIteration:
            self.init_loader()
            return self.loader.get_next_nb()
