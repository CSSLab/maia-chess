import haibrid_chess_utils

import argparse
import time
import humanize
import multiprocessing
import bz2
import io
import os
import os.path
import pandas
import numpy as np

#np.seterr('raise')

import csv

target_columns = [
    'num_ply',
    'winrate',
    'winrate_loss',
    'is_blunder_wr',
    'active_elo',
    'opponent_elo',
    'active_won',
    #'clock_percent',
    #'opp_clock_percent',
    #'is_capture',
    'cp',
    'cp_rel',
    'cp_loss',
]

@haibrid_chess_utils.logged_main
def main():
    parser = argparse.ArgumentParser(description='Group by board', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('inputs', nargs = '+', help='input CSVs')
    parser.add_argument('outputDir', help='output CSVs dir')
    parser.add_argument('--nrows', type=int, help='number of rows to read in, FOR TESTING', default = None)

    args = parser.parse_args()
    os.makedirs(args.outputDir, exist_ok=True)

    haibrid_chess_utils.printWithDate(f"Starting groupby on {','.join(args.inputs)} writing to {args.outputDir}")
    board_dicts = []
    #for p in args.inputs:
    #    board_dicts.append(get_boards_dict(p, args.outputDir))
    with multiprocessing.Manager() as manager, multiprocessing.Pool(len(args.inputs) + 1) as pool:
        sem = manager.Semaphore()
        q_m = manager.Queue()
        q_c = manager.Queue()
        q_b = manager.Queue()
        q_n = manager.Queue()
        q_t = manager.Queue()
        queues = {
            'semaphore' : sem,
            'maps' : q_m,
            'counts' : q_c,
            'blunders' : q_b,
            'nonblunders' : q_n,
            'target_columns' : q_t,
        }
        board_dicts_ret = pool.starmap_async(get_boards_dict, ((p, args.outputDir, queues, args.nrows) for p in args.inputs))
        final_fname = os.path.join(args.outputDir, 'all_months.csv.bz2')

        haibrid_chess_utils.printWithDate(f"Done all months, merging")

        merge_dicts(queues, board_dicts_ret, final_fname)

#. fen, num times observed, p(blunder), avg rating in position, avg opp rating, avg clock, avg opp clock, mode move played, mode blunder played

def merge_dicts(queues, rets, final_fname):
    dicts_lst = []
    counts_dict_com = {}
    maps_dict_com = {}
    blunder_moves_com = {}
    nonblunder_moves_com = {}
    target_columns_dicts_com = {c : {} for c in target_columns}
    count = 0
    while not rets.ready() or not queues['maps'].empty():
        if queues['maps'].empty():
            time.sleep(1)
        else:
            queues['semaphore'].acquire()
            maps_dict = queues['maps'].get()
            counts_dict = queues['counts'].get()
            blunder_dict = queues['blunders'].get()
            nonblunder_dict = queues['nonblunders'].get()
            columns_dict = queues['target_columns'].get()
            queues['semaphore'].release()
            count += 1
            haibrid_chess_utils.printWithDate(f"Merging {count}, {len(maps_dict)} total boards")
            for h_board, c in counts_dict.items():
                try:
                    counts_dict_com[h_board] += c
                except KeyError:
                    maps_dict_com[h_board] = maps_dict[h_board]
                    counts_dict_com[h_board] = c
                    blunder_moves_com[h_board] = blunder_dict[h_board]
                    nonblunder_moves_com[h_board] = nonblunder_dict[h_board]
                    for c in target_columns:
                        target_columns_dicts_com[c][h_board] = columns_dict[c][h_board]
                else:
                    blunder_moves_com[h_board] += blunder_dict[h_board]
                    nonblunder_moves_com[h_board] += nonblunder_dict[h_board]
                    for c in target_columns:
                        target_columns_dicts_com[c][h_board] += columns_dict[c][h_board]
    haibrid_chess_utils.printWithDate(f"Done merging, writing to {final_fname}")
    write_dicts(final_fname, maps_dict_com, counts_dict_com, blunder_moves_com, nonblunder_moves_com, target_columns_dicts_com)

#@profile
def get_boards_dict(fPath, outputDir, queues, nrows):
    name = os.path.basename(fPath).split('.')[0]
    haibrid_chess_utils.printWithDate(f"Starting: {name}")
    maps_dict = {}
    counts_dict = {}
    blunder_moves = {}
    nonblunder_moves = {}
    target_columns_dicts = {c : {} for c in target_columns}
    with bz2.open(fPath, 'rt') as f:
        reader = csv.DictReader(f)
        tstart = time.time()
        for i, line in enumerate(reader):
            #import pdb; pdb.set_trace()
            if line['low_time'] == 'True':
                continue
            if nrows is not None and i >= nrows:
                break
            #if i % 10000 == 0:
            #    haibrid_chess_utils.printWithDate(f"{name} {i} rows dones, {i /(time.time() - tstart):.2f} rows per second", end ='\r')
            #if i > 100000:
            #    break
            #import pdb; pdb.set_trace()
            board_s = ' '.join(line['board'].split(' ')[:2])

            #, colour, castling = haibrid_chess_utils.preproc_fen(line['board'])
            board_hash = hash(board_s)
            try:
                counts_dict[board_hash] += 1
            except KeyError:
                counts_dict[board_hash] = 1
                maps_dict[board_hash] = board_s
                if line['is_blunder_wr'] == 'True':
                    blunder_moves[board_hash] = [line['move']]
                    nonblunder_moves[board_hash] = []
                else:
                    nonblunder_moves[board_hash] = [line['move']]
                    blunder_moves[board_hash] = []
                for c in target_columns:
                    target_columns_dicts[c][board_hash] = [val_to_float(line[c])]
            else:
                if line['is_blunder_wr'] == 'True':
                    blunder_moves[board_hash].append(line['move'])
                else:
                    nonblunder_moves[board_hash].append(line['move'])
                for c in target_columns:
                    target_columns_dicts[c][board_hash].append(val_to_float(line[c]))
    haibrid_chess_utils.printWithDate(f"{name} done, now writing")
    #name = os.path.basename(fPath).split('.')[0]
    outputName = os.path.join(outputDir, f"{name}.csv.bz2")
    for k in list(maps_dict.keys()):
        if counts_dict[k] < 2:
            del maps_dict[k]
            del counts_dict[k]
            del blunder_moves[k]
            del nonblunder_moves[k]
            for c in target_columns:
                del target_columns_dicts[c][k]

    write_dicts(outputName, maps_dict, counts_dict, blunder_moves, nonblunder_moves, target_columns_dicts)
    sub_dicts = {i : {
                'maps' : {},
                'counts' : {},
                'blunders' : {},
                'nonblunders' : {},
                'target_columns' : {c : {} for c in target_columns},
            } for i in range(8)}

    for k in list(maps_dict.keys()):
        if counts_dict[k] < 2:
            pass
        else:
            s_d = sub_dicts[k % 8]
            s_d['maps'][k] = maps_dict[k]
            s_d['counts'][k] = counts_dict[k]
            s_d['blunders'][k] = blunder_moves[k]
            s_d['nonblunders'][k] = nonblunder_moves[k]
            for c in target_columns:
                s_d['target_columns'][c][k] = target_columns_dicts[c][k]

    for k in range(8):
        queues['semaphore'].acquire()
        queues['maps'].put(sub_dicts[k]['maps'])
        queues['counts'].put(sub_dicts[k]['counts'])
        queues['blunders'].put(sub_dicts[k]['blunders'])
        queues['nonblunders'].put(sub_dicts[k]['nonblunders'])
        queues['target_columns'].put(sub_dicts[k]['target_columns'])
        queues['semaphore'].release()

def val_to_float(v):
    try:
        return float(v)
    except ValueError:
        if v == 'True':
            return 1.
        elif v == 'False':
            return 0.
        return float('nan')

def mode(lst):
    try:
        return max(set(lst), key = lst.count)
    except ValueError:
        return ''

#@profile
def write_dicts(fname, maps_dict, counts_dict, blunder_dict, nonblunder_dict, columns_dict):
    with bz2.open(fname, 'wt') as f:
        header_vals = ['count', 'num_blunder', 'top_blunder', 'num_nonblunder', 'top_nonblunder']
        for c in target_columns:
            header_vals.append(f"{c}_mean")
            header_vals.append(f"{c}_var")
        f.write('board,')
        f.write(','.join(header_vals) + '\n')
        for h_board, c in counts_dict.items():
            if c <= 1:
                continue
            b_list = blunder_dict[h_board]
            nb_list = nonblunder_dict[h_board]
            row_vals = [
                    maps_dict[h_board],
                    c,
                    len(b_list),
                    mode(b_list),
                    len(nb_list),
                    mode(nb_list),
                    ]
            #np.seterr(all='raise', divide = 'raise')
            for c in target_columns:
                c_vals = columns_dict[c][h_board]
                row_vals.append(np.nanmean(c_vals))
                row_vals.append(np.nanvar(c_vals))
            f.write(','.join([str(v) for v in row_vals]) + '\n')

if __name__ == '__main__':
    main()
