import maia_chess_backend

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
    'move_ply',
    'cp_rel',
    'cp_loss',
    'is_blunder_cp',
    'winrate',
    'winrate_elo',
    'winrate_loss',
    'is_blunder_wr',
    'opp_winrate',
    'white_active',
    'active_elo',
    'opponent_elo',
    'clock_percent',
    'opp_clock_percent',
    'is_capture',
    'is_check',
    'active_won',
    'no_winner',
    'num_ply'
]


target_columns =  mmap_columns + ['game_id', 'low_time', 'board', 'move']

def main():
    parser = argparse.ArgumentParser(description='Make mmapped version of csv', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('inputs', nargs = '+', help='input csv')
    parser.add_argument('outputDir', help='output dir of mmapped files')
    parser.add_argument('--nrows', type=int, help='number of rows to read in, FOR TESTING', default = None)

    parser.add_argument('--min_elo', type=int, help='min active elo', default = 1000)
    parser.add_argument('--max_elo', type=int, help='max active elo', default = 4000)
    parser.add_argument('--allow_negative_loss', type=bool, help='allow winrate losses below 0', default = False)
    parser.add_argument('--allow_low_time', type=bool, help='Include low time moves', default = False)
    parser.add_argument('--min_ply', type=int, help='min move ply to consider', default = 6)

    parser.add_argument('--nb_to_b_ratio', type=float, help='ratio fof blunders to non blunders in dataset', default = 1.5)


    #parser.add_argument('split_column', help='what to split the csvs on, i.e. is_blunder')

    #parser.add_argument('y_vals', nargs = '+', help='columns to treate as y vals')

    args = parser.parse_args()

    maia_chess_backend.printWithDate(f"Starting mmap of {', '.join(args.inputs)} writing to {args.outputDir} with {', '.join(mmap_columns)}")

    with  multiprocessing.Pool(32) as pool:
        pool.starmap(run_path, [(p, args) for p in args.inputs])
    maia_chess_backend.printWithDate("Done")

def run_path(path, args):
    #helper for multiproc
    try:
        mmap_csv(
                path,
                load_csv(path, args.nrows),
                args.outputDir,
                args,
            )
    except EOFError:
        maia_chess_backend.printWithDate(f"EOF error in: {path}")

def load_csv(target_path, nrows):
    maia_chess_backend.printWithDate(f"Loading: {target_path}", flush = True)
    return pandas.read_csv(target_path, usecols=target_columns, nrows = nrows)

def mmap_csv(target_path, df, outputDir, args):
    maia_chess_backend.printWithDate(f"Loading: {target_path}")
    name = os.path.basename(target_path).split('.')[0]

    #df =

    maia_chess_backend.printWithDate(f"Filtering data starting at {len(df)} rows")

    df = df[df['move_ply'] >= args.min_ply]

    if not args.allow_low_time:
        df = df[df['low_time'].eq(False)]

    if not args.allow_negative_loss:
        df = df[df['winrate_loss'] > 0]

    df = df[df['active_elo'] > args.min_elo]
    df = df[df['active_elo'] < args.max_elo]

    df = df.dropna()

    maia_chess_backend.printWithDate(f"Filtering down data to {len(df)} rows")

    df_blunder = df[df['is_blunder_wr']]
    maia_chess_backend.printWithDate(f"Found {len(df_blunder)} blunders")

    df_blunder = df_blunder.sample(frac=1).reset_index(drop=True)

    df_non_blunder = df[df['is_blunder_wr'].eq(False)]
    maia_chess_backend.printWithDate(f"Found {len(df_non_blunder)} non blunders")

    df_non_blunder = df_non_blunder.sample(frac=1).reset_index(drop=True).iloc[:int(len(df_blunder) * args.nb_to_b_ratio)]

    del df

    maia_chess_backend.printWithDate(f"Reduced to {len(df_non_blunder)} non blunders")

    maia_chess_backend.printWithDate(f"Starting mmaping")

    os.makedirs(outputDir, exist_ok = True)
    make_df_mmaps(df_blunder, name, os.path.join(outputDir, name, 'blunder'))
    del df_blunder
    make_df_mmaps(df_non_blunder, name, os.path.join(outputDir, name, 'nonblunder'))

def make_var_mmap(y_name, outputPath, mmaps, df):
    a_c = df[y_name].values
    if a_c.dtype == np.bool:
        a_c = a_c.astype(np.long)
    else:
        a_c = a_c.astype(np.float32)
    mmaps[y_name] = np.memmap(
            os.path.join(outputPath, f"{y_name}+{a_c.dtype}+{a_c.shape[0]}.mm"),
            dtype = a_c.dtype,
            mode = 'w+',
            shape = a_c.shape,
            )
    mmaps[y_name][:] = a_c[:]

def make_board_mmap(outputPath, mmaps, df):
    b_sample_shape = maia_chess_backend.fenToVec(chess.Board().fen()).shape

    mmap_vec = np.memmap(
        os.path.join(outputPath, f"board+{len(df)}.mm"),
        dtype=np.bool,
        mode='w+',
        shape=(len(df), b_sample_shape[0], b_sample_shape[1], b_sample_shape[2]),
        )
    for i, (_, row) in enumerate(df.iterrows()):
        mmap_vec[i, :] = maia_chess_backend.fenToVec(row['board'])[:]
    #a_boards = np.stack(pool.map(maia_chess_backend.fenToVec, df['board']))
    mmaps['board'] = mmap_vec

def make_move_mmap(outputPath, mmaps, df):
    a_moves = np.stack(df['move'].apply(maia_chess_backend.move_to_index))
    mmaps['move'] = np.memmap(
            os.path.join(outputPath, f"move+{a_moves.shape[0]}.mm"),
            dtype=a_moves.dtype,
            mode='w+',
            shape=a_moves.shape,
            )
    mmaps['move'][:] = a_moves[:]

def make_game_id_mmap(outputPath, mmaps, df):
    game_ids = set(df['game_id'])
    game_ids_dict = {i : g_id for i, g_id in enumerate(game_ids)}
    game_ids_reverse_dict = {g_id : i for i, g_id in game_ids_dict.items()}
    with open(os.path.join(outputPath, "game_id_lookup.json"), 'w') as f:
        json.dump(game_ids_dict, f, indent=2)

    a_c = df['game_id'].apply(lambda x : game_ids_reverse_dict[x]).values
    a_c = a_c.astype(np.long)
    mmaps['game_id'] = np.memmap(
            os.path.join(outputPath, f"game_id+{a_c.dtype}+{a_c.shape[0]}.mm"),
            dtype=a_c.dtype,
            mode='w+',
            shape=a_c.shape,
            )
    mmaps['game_id'][:] = a_c[:]
def make_df_mmaps(df, name, output_dir):
    os.makedirs(output_dir, exist_ok = True)

    mmaps = {}
    maia_chess_backend.printWithDate(f"Making y_vals mmaps for: {name}", flush = True)
    for y_name in mmap_columns:
        make_var_mmap(y_name, output_dir, mmaps, df)
        #print(y_name, end = ' ', flush = True)

    make_game_id_mmap(output_dir, mmaps, df)

    maia_chess_backend.printWithDate(f"Making move array mmaps for: {name}", flush = True)
    make_move_mmap(output_dir, mmaps, df)

    maia_chess_backend.printWithDate(f"Making boards array mmaps for: {name}", flush = True)
    make_board_mmap(output_dir, mmaps, df)


if __name__ == '__main__':
    main()
