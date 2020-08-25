import argparse
import time
import humanize
import multiprocessing
import bz2
import io
import os
import os.path
import re
import queue
import zipfile
import pandas

import numpy as np

import chess
import chess.pgn

import haibrid_chess_utils

target_columns = [
    'game_id',
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
    'active_won',
    'low_time',
    'board',
]

@haibrid_chess_utils.logged_main
def main():
    parser = argparse.ArgumentParser(description='Create two new csvs with select columns split by is_blunder_wr', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('input', help='input CSV')
    parser.add_argument('outputDir', help='output CSV')

    parser.add_argument('--min_elo', type=int, help='min active elo', default = 1000)
    parser.add_argument('--max_elo', type=int, help='min active elo', default = 9999999999)
    parser.add_argument('--allow_negative_loss', type=bool, help='allow winrate losses below 0', default = False)
    parser.add_argument('--allow_low_time', type=bool, help='Include low time moves', default = False)

    parser.add_argument('--min_ply', type=int, help='min move ply to consider', default = 6)

    #parser.add_argument('--shuffleSize', type=int, help='Shuffle buffer size', default = 1000)
    parser.add_argument('--nrows', type=int, help='number of rows to read in', default = None)

    parser.add_argument('--nb_to_b_ratio', type=float, help='ratio fof blunders to non blunders in dataset', default = 1.5)

    args = parser.parse_args()

    haibrid_chess_utils.printWithDate(f"Starting CSV split of {args.input} writing to {args.outputDir}")
    haibrid_chess_utils.printWithDate(f"Collecting {', '.join(target_columns)}")

    name = os.path.basename(args.input).split('.')[0]
    outputBlunder = os.path.join(args.outputDir, f"{name}_blunder.csv.bz2")
    outputNonBlunder = os.path.join(args.outputDir, f"{name}_nonblunder.csv.bz2")

    haibrid_chess_utils.printWithDate(f"Created outputs named {outputBlunder} and {outputNonBlunder}")



    os.makedirs(args.outputDir, exist_ok = True)

    haibrid_chess_utils.printWithDate(f"Starting read")
    with bz2.open(args.input, 'rt') as f:
        df = pandas.read_csv(f, usecols = target_columns, nrows = args.nrows)


    haibrid_chess_utils.printWithDate(f"Filtering data starting at {len(df)} rows")

    df = df[df['move_ply'] >= args.min_ply]

    if not args.allow_low_time:
        df = df[df['low_time'].eq(False)]

    if not args.allow_negative_loss:
        df = df[df['winrate_loss'] > 0]

    df = df[df['active_elo'] > args.min_elo]
    df = df[df['active_elo'] < args.max_elo]

    df = df.dropna()

    haibrid_chess_utils.printWithDate(f"Filtering down data to {len(df)} rows")

    df_blunder = df[df['is_blunder_wr']]
    haibrid_chess_utils.printWithDate(f"Found {len(df_blunder)} blunders")

    df_blunder = df_blunder.sample(frac=1).reset_index(drop=True)

    df_non_blunder = df[df['is_blunder_wr'].eq(False)]
    haibrid_chess_utils.printWithDate(f"Found {len(df_non_blunder)} non blunders")

    df_non_blunder = df_non_blunder.sample(frac=1).reset_index(drop=True).iloc[:int(len(df_blunder) * args.nb_to_b_ratio)]

    haibrid_chess_utils.printWithDate(f"Reduced to {len(df_non_blunder)} non blunders")

    haibrid_chess_utils.printWithDate(f"Starting writing")

    with bz2.open(outputNonBlunder, 'wt') as fnb:
        df_non_blunder.to_csv(fnb, index = False)
    with bz2.open(outputBlunder, 'wt') as fb:
        df_blunder.to_csv(fb, index = False)

if __name__ == '__main__':
    main()
