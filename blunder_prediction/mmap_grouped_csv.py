import haibrid_chess_utils

import bz2
import argparse
import os
import os.path
import multiprocessing
import time
import json
import chess

import numpy as np
import pandas

mmap_columns = [
    'is_blunder_mean',
    'is_blunder_wr_mean',
    'active_elo_mean',
    'opponent_elo_mean',
    'active_won_mean',
    'cp_rel_mean',
    'cp_loss_mean',
    'num_ply_mean',
]


target_columns =  mmap_columns + ['board_extended', 'top_nonblunder', 'top_blunder']

def main():
    parser = argparse.ArgumentParser(description='Make mmapped version of csv', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('inputs', nargs = '+', help='input csv')
    parser.add_argument('outputDir', help='output dir of mmapped files')
    parser.add_argument('--nrows', type=int, help='number of rows to read in, FOR TESTING', default = None)

    args = parser.parse_args()

    haibrid_chess_utils.printWithDate(f"Starting mmap of {', '.join(args.inputs)} writing to {args.outputDir} with {', '.join(mmap_columns)}")

    mmaps = {}
    for path in args.inputs:
        mmaps[path] = mmap_csv(
        path,
        load_csv(path, args.nrows),
        args.outputDir,
        args,
        )
    haibrid_chess_utils.printWithDate("All mmapped")
    try:
        while True:
            haibrid_chess_utils.printWithDate("Still alive", end = '\r')
            time.sleep(10 * 60)
    except KeyboardInterrupt:
        print()
        haibrid_chess_utils.printWithDate("Exiting")

def load_csv(target_path, nrows):
    haibrid_chess_utils.printWithDate(f"Loading: {target_path}", flush = True)
    return pandas.read_csv(target_path, usecols=target_columns, nrows = nrows)

def mmap_csv(target_path, df, outputDir, args):
    haibrid_chess_utils.printWithDate(f"Loading: {target_path}")
    name = os.path.basename(target_path).split('.')[0]

    df_blunder = df[df['is_blunder_mean']]
    haibrid_chess_utils.printWithDate(f"Found {len(df_blunder)} blunders")

    df_blunder = df_blunder.sample(frac=1).reset_index(drop=True)

    df_non_blunder = df[df['is_blunder_mean'].eq(False)]
    haibrid_chess_utils.printWithDate(f"Found {len(df_non_blunder)} non blunders")

    df_non_blunder = df_non_blunder.sample(frac=1).reset_index(drop=True)

    del df

    haibrid_chess_utils.printWithDate(f"Reduced to {len(df_non_blunder)} non blunders")

    haibrid_chess_utils.printWithDate(f"Starting mmaping")

    os.makedirs(outputDir, exist_ok = True)

    mmaps = {}

    mmaps['blunder'] = make_df_mmaps(df_blunder, name, os.path.join(outputDir, 'blunder'))

    del df_blunder
    mmaps['nonblunder'] = make_df_mmaps(df_non_blunder, name, os.path.join(outputDir, 'nonblunder'))
    return mmaps

def make_var_mmap(y_name, outputName, mmaps, df):
    a_c = df[y_name].values
    if a_c.dtype == np.bool:
        a_c = a_c.astype(np.long)
    else:
        a_c = a_c.astype(np.float32)
    mmaps[y_name] = np.memmap(f"{outputName}+{y_name}+{a_c.dtype}+{a_c.shape[0]}.mm", dtype=a_c.dtype, mode='w+', shape=a_c.shape)
    mmaps[y_name][:] = a_c[:]

def make_board_mmap(outputName, mmaps, df):

    b_sample_shape = haibrid_chess_utils.fenToVec(chess.Board().fen()).shape

    mmap_vec = np.memmap(
        f"{outputName}+board+{len(df)}.mm",
        dtype=np.bool,
        mode='w+',
        shape=(len(df), b_sample_shape[0], b_sample_shape[1], b_sample_shape[2]),
        )
    for i, (_, row) in enumerate(df.iterrows()):
        mmap_vec[i, :] = haibrid_chess_utils.fenToVec(row['board_extended'])[:]
    #a_boards = np.stack(pool.map(haibrid_chess_utils.fenToVec, df['board']))
    mmaps['board'] = mmap_vec

def move_index_nansafe(move):
    try:
        return haibrid_chess_utils.move_to_index(move)
    except TypeError:
        return -1

def make_move_mmap(outputName, mmaps, moves_name, df):
    a_moves = np.stack(df[moves_name].apply(move_index_nansafe))
    mmaps[moves_name] = np.memmap(f"{outputName}+{moves_name}+{a_moves.shape[0]}.mm", dtype=a_moves.dtype, mode='w+', shape=a_moves.shape)
    mmaps[moves_name][:] = a_moves[:]

def make_df_mmaps(df, name, output_dir):
    os.makedirs(output_dir, exist_ok = True)
    outputName = os.path.join(output_dir, name)

    mmaps = {}
    haibrid_chess_utils.printWithDate(f"Making y_vals mmaps for: {name} done:", end = ' ')
    for y_name in mmap_columns:
        make_var_mmap(y_name, outputName, mmaps, df)
        print(y_name, end = ' ', flush = True)

    haibrid_chess_utils.printWithDate(f"Making move array mmaps for: {name}")

    make_move_mmap(outputName, mmaps, 'top_blunder', df)
    make_move_mmap(outputName, mmaps, 'top_nonblunder', df)

    haibrid_chess_utils.printWithDate(f"Making boards array mmaps for: {name}")

    make_board_mmap(outputName, mmaps, df)

    return mmaps

if __name__ == '__main__':
    main()
