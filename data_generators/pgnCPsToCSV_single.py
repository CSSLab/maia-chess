#Most of the functions are imported from multi
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

import chess
import chess.pgn

import haibrid_chess_utils

from pgnCPsToCSV_multi import *

def cleanup(pgnReaders, gameReaders, writers):
    pgnReaders.get()
    haibrid_chess_utils.printWithDate(f"Done reading")
    time.sleep(10)
    for r in gameReaders:
        r.get()
    haibrid_chess_utils.printWithDate(f"Done processing")

    writers.get()

@haibrid_chess_utils.logged_main
def main():
    parser = argparse.ArgumentParser(description='process PGN file with stockfish annotaions into a csv file', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('input', help='input PGNs')
    parser.add_argument('outputDir', help='output CSVs dir')

    parser.add_argument('--pool', type=int, help='number of simultaneous jobs running per fil', default = 30)
    #parser.add_argument('--readers', type=int, help='number of simultaneous reader running per inputfile', default = 24)
    parser.add_argument('--queueSize', type=int, help='Max number of games to cache', default = 1000)

    args = parser.parse_args()

    haibrid_chess_utils.printWithDate(f"Starting CSV conversion of {args.input} writing to {args.outputDir}")

    os.makedirs(args.outputDir, exist_ok=True)

    name = os.path.basename(args.input).split('.')[0]
    outputName = os.path.join(args.outputDir, f"{name}.csv.bz2")
    #names[n] = (name, outputName)


    haibrid_chess_utils.printWithDate(f"Loading file: {name}")
    haibrid_chess_utils.printWithDate(f"Starting main loop")

    tstart = time.time()

    print(args)
    with multiprocessing.Manager() as manager:
        with multiprocessing.Pool(args.pool) as workers_pool, multiprocessing.Pool(3) as io_pool:
            pgnReader, gameReader, writer, unproccessedQueue, resultsQueue = processPGN(args.input, name, outputName, args.queueSize, args.pool, manager, workers_pool, io_pool)

            haibrid_chess_utils.printWithDate(f"Done loading Queues in {humanize.naturaldelta(time.time() - tstart)}, waiting for reading to finish")

            cleanup(pgnReader, gameReader, writer)

    haibrid_chess_utils.printWithDate(f"Done everything in {humanize.naturaldelta(time.time() - tstart)}, exiting")

if __name__ == '__main__':
    main()
