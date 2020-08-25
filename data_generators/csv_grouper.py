import json
import argparse
import time
import os
import os.path
import humanize
import multiprocessing
import bz2
import io
import re
import queue
import pandas
import numpy as np

import chess
import chess.pgn

import haibrid_chess_utils

import haibrid_chess_utils.pickle4reducer
ctx = multiprocessing.get_context()
ctx.reducer = haibrid_chess_utils.pickle4reducer.Pickle4Reducer()

groupbyColumns = {
    'roundedCP' : 'binned_cp',
    'CP' : 'cp_rel',
    'ELO' : 'binned_active_elo',
    'oppELO' : 'binned_opponent_elo',
    'rawELO' : 'active_elo',
    'type' : 'type',
    'clock' : 'binned_clock',
    'oppClock' : 'opp_binned_clock',
    'phase' : 'game_phase',
    'lowTime' : 'low_time',
    'delta' : 'elo_delta'
}

time_cutoffs = [5, 10, 30, 60, 120, 300, 600]
low_time = 60

def isLowTime(ct):
    return True if ct < low_time else False

def binClocks(ct):
    for c in time_cutoffs:
        if ct < c:
            return c
    return 1000

def binPly(move_num):
    if move_num < 20:
        return 'early'
    elif move_num < 100:
        return 'mid'
    return 'late'

operations = [
    ['lowTime', 'roundedCP'],
    ['lowTime', 'roundedCP', 'ELO'],
    ['lowTime', 'roundedCP', 'oppELO'],
    ['lowTime', 'roundedCP', 'ELO', 'oppELO'],
    ['lowTime', 'roundedCP', 'ELO', 'phase'],
    ['lowTime', 'roundedCP', 'ELO', 'type'],
    ['lowTime', 'roundedCP', 'ELO', 'clock'],
    ['lowTime', 'roundedCP', 'delta'],
    ['lowTime', 'roundedCP', 'ELO', 'delta'],
    ['lowTime', 'roundedCP', 'ELO', 'oppClock'],
    ['lowTime', 'roundedCP', 'ELO', 'oppClock', 'oppELO'],
    ['lowTime', 'roundedCP', 'ELO', 'clock', 'oppClock'],
    #['roundedCP', 'ELO', 'phase', 'type'],
    #['roundedCP', 'ELO', 'phase', 'clock'],
    #['roundedCP', 'ELO', 'phase', 'clock', 'type'],
    #['CP', 'ELO'],
    #['CP', 'ELO', 'type'],
    #['roundedCP', 'rawELO'],
    #['roundedCP', 'rawELO', 'type'],
    #['CP', 'rawELO'],
    #['CP', 'rawELO', 'type'],
]


def runGroupBys(df_working, outputDir, fname):
    for opList in operations:
        mStart = time.time()
        opgroups = [groupbyColumns[n] for n in opList]

        opName = 'on-' + '_'.join(opList)

        df_c = df_working.groupby(opgroups).count().reset_index()
        df_m = df_working.groupby(opgroups).mean().reset_index()
        df_m['count'] = df_c['clock']

        df_m.to_csv(os.path.join(outputDir, f"{fname}_{opName}.csv"))

        haibrid_chess_utils.printWithDate(f"{fname} groupby: {opName} done in {humanize.naturaldelta(time.time() - mStart)}", colour = 'pink', flush = True)

def processTarget(target, outputDir, nrows, queue_out):
    tstart = time.time()
    os.makedirs(outputDir, exist_ok = True)
    tname = os.path.basename(target).split('.')[0]
    haibrid_chess_utils.printWithDate(f"Starting on {tname}", colour = 'green', flush = True)

    with bz2.open(target, 'rt') as f:
        df = pandas.read_csv(f, index_col = None, nrows = nrows, usecols = ['cp_rel', 'active_elo' , 'opponent_elo', 'clock', 'opp_clock', 'move_ply', 'type', 'low_time', 'active_won'])

    haibrid_chess_utils.printWithDate(f"{tname} loaded: {len(df)} lines in {humanize.naturaldelta(time.time() - tstart)}", colour = 'green', flush = True)

    #df['mul_factor'] = df['white_active'].apply(lambda x : 1 if x else -1)
    #df['cp_rel'] = df['cp'] * df['mul_factor']

    df['binned_cp'] = df['cp_rel'].apply(lambda x : x * 100 // 10 / 10)
    df['binned_active_elo'] = df['active_elo'].apply(lambda x : (x // 100) * 100)
    df['binned_opponent_elo'] = df['opponent_elo'].apply(lambda x : (x // 100) * 100)

    df['elo_delta'] = (df['active_elo'] - df['opponent_elo']).apply(lambda x : (x // 100) * 100)

    df['binned_clock'] = df['clock'].apply(binClocks)
    df['opp_binned_clock'] = df['opp_clock'].apply(binClocks)
    df['game_phase'] = df['move_ply'].apply(binPly)
    #df['low_time'] = df['clock'].apply(isLowTime)

    for df_sub in np.split(df, range(10**6, len(df), 10**6), axis = 0):
        queue_out.put(df_sub.copy())

    queue_out.put("done")

    runGroupBys(df, outputDir, tname)

    haibrid_chess_utils.printWithDate(f"{tname} done everything in {humanize.naturaldelta(time.time() - tstart)}", colour = 'green', flush = True)

def main():
    parser = argparse.ArgumentParser(description='Run groupby on the csv files', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('inputs', nargs = '+', help='input CSVs')
    parser.add_argument('outputDir', help='output dir name')

    parser.add_argument('--pool', type=int, help='number of simultaneous jobs running', default = 32)

    parser.add_argument('--nrows', type=int, help='For debugging, limit the number lines read', default = None)

    args = parser.parse_args()

    haibrid_chess_utils.printWithDate(f"Starting CSV groupby of {', '.join(args.inputs)} writing to {args.outputDir}")

    os.makedirs(args.outputDir, exist_ok = True)

    tstart = time.time()

    with multiprocessing.Manager() as manager:
        dfs_queue = manager.Queue()

        with multiprocessing.Pool(args.pool) as pool:
            mResult = pool.starmap_async(processTarget, [(tname, os.path.join(args.outputDir, os.path.basename(tname).split('.')[0]), args.nrows, dfs_queue) for tname in args.inputs])

            dfs = []
            dones = []
            while len(dones) < len(args.inputs):
                try:
                    r = dfs_queue.get(True, 10)
                    if isinstance(r, str):
                        dones.append(r)
                    else:
                        dfs.append(r)
                except queue.Empty:
                    pass

            tname = "lichess_db_standard_rated_all"

            haibrid_chess_utils.printWithDate(f"All {len(dfs)} dfs collected")
            df = pandas.concat(dfs, ignore_index=True)

            haibrid_chess_utils.printWithDate(f"Full dataset loaded: {len(df)} lines")

            runGroupBys(df, args.outputDir, tname)

            mJoined = mResult.get()
            haibrid_chess_utils.printWithDate(f"Done all single months in {humanize.naturaldelta(time.time() - tstart)}")

    haibrid_chess_utils.printWithDate(f"{tname} done everything in {humanize.naturaldelta(time.time() - tstart)}, exiting")

if __name__ == '__main__':
    main()
