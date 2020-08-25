import pandas
import numpy as np
import zipfile
import io
import os
import os.path
import random
import multiprocessing
import bz2

from .fen_to_vec import fenToVec

class CSVLoader(object):
    def __init__(self, blunders_fname, non_blunders_fname, y_names, mini_batch_size = 500, nrows = None):
        self.blunders_fname = blunders_fname
        self.non_blunders_fname = non_blunders_fname
        self.mini_batch_size = mini_batch_size
        self.y_names = y_names
        self.columns = self.y_names + ['board', 'is_blunder_wr']

        with bz2.open(self.blunders_fname) as f:
            self.df_blunders = pandas.read_csv(f, usecols = self.columns, nrows = nrows).sample(frac=1).reset_index(drop=True)

        with bz2.open(self.non_blunders_fname) as f:
            self.df_nonblunders = pandas.read_csv(f, usecols = self.columns, nrows = nrows).sample(frac=1).reset_index(drop=True)

        self.nb_index = 0
        self.nb_max = len(self.df_nonblunders)
        self.b_index = 0
        self.b_max = len(self.df_blunders)
        self.batch_stride = self.mini_batch_size // 2

    def __len__(self):
        return len(self.df_blunders) + len(self.df_nonblunders)

    def process_rows(self, df_sub):
        x = list(df_sub['board'].apply(fenToVec))
        y_vals = {}
        for n in self.y_names:
            y_vals[n] = list(df_sub[n])
        return x, y_vals

    def next_blunder(self):
        b_index_old = self.b_index
        self.b_index += self.batch_stride
        if self.b_index >= self.b_max:
            self.b_index = 0
            self.df_blunders = self.df_blunders.sample(frac=1).reset_index(drop=True)
            return self.next_blunder()
        return self.process_rows(self.df_blunders.iloc[b_index_old:self.b_index])

    def next_nonblunder(self):
        nb_index_old = self.nb_index
        self.nb_index += self.batch_stride
        if self.nb_index >= self.nb_max:
            self.nb_index = 0
            self.df_nonblunders = self.df_nonblunders.sample(frac=1).reset_index(drop=True)
            return self.next_nonblunder()
        return self.process_rows(self.df_nonblunders.iloc[nb_index_old:self.nb_index])

    def __iter__(self):
        while True:

            x_nb, ys_nb = self.next_nonblunder()
            x_b, ys_b = self.next_blunder()

            x_batch = x_nb + x_b
            ys_batch = {}
            for n in self.y_names:
                ys_batch[n] = ys_nb[n] + ys_b[n]
            y_dict = {}
            for n, a in ys_batch.items():
                a_c = np.array(a)
                if a_c.dtype == np.bool:
                    a_c = a_c.astype(np.long)
                else:
                    a_c = a_c.astype(np.float32).reshape(-1, 1)
                y_dict[n] = a_c

            x_c = np.stack(x_batch, axis = 0).astype(np.float32)
            yield x_c, y_dict

class BinsDir(object):
    def __init__(self, path):
        self.path = path

    def filelist(self):
        files = []
        for i, file in enumerate(os.scandir(self.path)):
            files.append(file.name)
        return files

    def open(self, fname):
        return open(os.path.join(self.path, fname), 'rb')

    def close(self):
        pass

class BinFile(object):
    def __init__(self, path):
        self.path = path
        if os.path.isdir(path):
            self.handle = BinsDir(path)
            self.names = list(set([s[:-4] for s in self.handle.filelist()]))
            #print(f"bin {len(self.names)} names", flush=True)
        else:
            self.handle = zipfile.ZipFile(path)
            self.names = list(set([s.filename[:-4] for s in self.handle.filelist]))
        self._len = len(self.names)
        #print(f"bin {path} done", flush=True)

    def __len__(self):
        return self._len

    def __getitem__(self, key):
        try:
            with self.handle.open(f"{key}.npy") as f:
                arr = np.load(io.BytesIO(f.read()))
            with self.handle.open(f"{key}.csv") as f:
                df = pandas.read_csv(f)
        except FileNotFoundError:
            raise KeyError(f"{key} not found in zipfile")
        return (arr[:len(df)], df)

    def __del__(self):
        try:
            self.handle.close()
        except:
            pass

    def get_randomxy(self, ratio = None, key = None):
        #print(f"bin getting {self.path} entry", flush=True)
        if key is None:
            key = self.names[np.random.randint(self._len)]
        #print(f"bin {self.path} got key", flush=True)
        a, df = self[key]
        if ratio is None:
            i = random.randrange(len(df))
        elif np.random.random() < ratio:
            try:
                i = np.random.choice(df[df['blunder_wr'] == True].index)
            except (IndexError, ValueError):
                i = random.randrange(len(df))
        else:
            i = np.random.choice(df[df['blunder_wr'] == False].index)
        #print(f"bin {self.path} got index", flush=True)
        return a[i,:,:,:], int(df.iloc[i]['blunder_wr'])

    def keys(self):
        return self.names

def binLoader(target_file, working_queue):
    pass

class Binloader(object):
    def __init__(self, path, pool_size, queue_size):
        self.pool_size = pool_size
        self.pool = multiprocessing.Pool(pool_size)
        self.manager = multiprocessing.Manager()
        self.queue = self.manager.Queue(queue_size)
        self.path = path

        readers = []
        for _ in range(pool_size):
            self.pool.apply_async(binLoader, (path, self.queue))

class BatchFile(object):
    def __init__(self, path, y_names = None, mini_batch_size = 10):
        if y_names is None:
            y_names = ['winrate', 'winrate_loss', 'is_blunder_wr', 'active_won']
        self.path = path
        self.y_names = y_names
        self.mini_batch_size = mini_batch_size
        self.handle = zipfile.ZipFile(path)
        self.names_blunder = list(set([s.filename.split('_')[0] for s in self.handle.filelist if '_b_' in s.filename]))
        self.names_nonblunder = list(set([s.filename.split('_')[0] for s in self.handle.filelist if '_nb_' in s.filename]))
        self._len = len(self.names_blunder) + len(self.names_nonblunder)

        self.targets_blunder = list(self.names_blunder)
        np.random.shuffle(self.targets_blunder)
        self.targets_nonblunder = list(self.names_nonblunder)
        np.random.shuffle(self.targets_nonblunder)

        #self.batch_buffer_x = np.zeros((0,13, 8, 8))
        #self.batch_buffer_ys = [np.zeros(0) for i in range(4)]

    def __iter__(self):
        while True:
            x_batch = []
            ys_batch = {n : [] for n in self.y_names}
            for i in range(self.mini_batch_size // 2):
                (x_nb, ys_nb), (x_b, ys_b) = self.get_next()
                x_batch.append(x_b)
                x_batch.append(x_nb)
                for n in self.y_names:
                    ys_batch[n].append(ys_nb[n])
                    ys_batch[n].append(ys_b[n])

            y_dict = {}
            for n, a in ys_batch.items():
                a_c = np.concatenate(a, axis = 0)
                if a_c.dtype == np.bool:
                    a_c = a_c.astype(np.long)
                else:
                    a_c = a_c.astype(np.float32).reshape(-1, 1)
                y_dict[n] = a_c

            x_c = np.concatenate(x_batch, axis = 0).astype(np.float32)

            yield x_c, y_dict

    def get_next(self):
        if len(self.targets_blunder) < 1:
            self.targets_blunder = list(self.names_blunder)
            np.random.shuffle(self.targets_blunder)
        if len(self.targets_nonblunder) < 1:
            self.targets_nonblunder = list(self.names_nonblunder)
            np.random.shuffle(self.targets_nonblunder)
        return self.get_nonblunder(self.targets_nonblunder.pop()), self.get_blunder(self.targets_blunder.pop())

    def __len__(self):
        return self._len

    def __getitem__(self, key):
        with self.handle.open(f"{key}_x.npy") as f:
            x = np.load(io.BytesIO(f.read()))

        with self.handle.open(f"{key}_dat.csv") as f:
            df = pandas.read_csv(io.BytesIO(f.read()), usecols = self.y_names)

        y_dict = {n : np.array(df[n]) for n in self.y_names}

        return x, y_dict

    def get_nonblunder(self, key):
        return self[key + '_nb']

    def get_blunder(self, key):
        return self[key + '_b']


#Still might want to use this
#Yes this is in VC
class BatchFile_old(object):
    def __init__(self, path, multi_head = False):
        self.path = path
        self.handle = zipfile.ZipFile(path)
        self.names = list(set([s.filename.split('_')[0] for s in self.handle.filelist]))
        self._len = len(self.names)
        self.targets = list(self.names)
        np.random.shuffle(self.targets)
        self.multi_head = multi_head

    def __iter__(self):
        while True:
            if len(self.targets) < 1:
                self.targets = list(self.names)
                np.random.shuffle(self.targets)
            yield self[self.targets.pop()]

    def __len__(self):
        return self._len

    def __getitem__(self, key):
        with self.handle.open(f"{key}_x.npy") as f:
            x = np.load(io.BytesIO(f.read()))

        #with self.handle.open(f"{key}_y.npy") as f:
        #    y = np.load(io.BytesIO(f.read()))

        with self.handle.open(f"{key}_dat.csv") as f:
            df = pandas.read_csv(io.BytesIO(f.read()))
        if not self.multi_head:
            return x, np.array(df['blunder_wr'])
        else:
            return x, np.array(df['winrate']), np.array(df['winrate_loss']), np.array(df['blunder_wr']), np.array(df['active_won'])

    def get_dat(self, key):
        with self.handle.open(f"{key}_dat.csv") as f:
            return pandas.read_csv(io.BytesIO(f.read()))
