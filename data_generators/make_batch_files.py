# similar to pgnCPsToCSV_single but makes a zip with make seperate csvs
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

import numpy as np

import chess
import chess.pgn

import haibrid_chess_utils

logging_delay = 30 # in seconds
game_per_put = 5

elow_re = re.compile(r'\[WhiteElo "(\d+)"\]')
elob_re = re.compile(r'\[BlackElo "(\d+)"\]')

per_game_header =  [
    'game_id',
]

per_move_header = [
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
]
parseHeader = per_game_header + per_move_header

def weighted_sort(a, var_factor = 2):
    w = [np.random.normal(i, len(a)/var_factor) for i in range(len(a))]
    return [i for _, i in sorted(list(zip(w, a)), reverse = True)]

def parseLines(gameStr, num_samples, min_elo, max_elo, allow_negative_loss, allow_low_time):
    try:
        elo_w = int(elow_re.search(gameStr).group(1))
        elo_b = int(elob_re.search(gameStr).group(1))
    except:
        return [], []
    if (elo_w > max_elo) or (elo_w < min_elo):
        return [], []
    if (elo_b > max_elo) or (elo_w < min_elo):
        return [], []
    try:
        df = haibrid_chess_utils.gameToDF(
                    gameStr,
                    per_game_header,
                    per_move_header,
                    with_board_stats = False,
                    )
    except haibrid_chess_utils.NoStockfishEvals:
        return [], []
    if not allow_low_time:
        df = df[df['low_time'].eq(False)]
    if not allow_negative_loss:
        df = df[df['winrate_loss'] > 0]

    game = chess.pgn.read_game(io.StringIO(gameStr))
    boards_arr = haibrid_chess_utils.gameToVecs(game)

    df_blunders = df[df['is_blunder_wr']]
    df_non_blunders = df[df['is_blunder_wr'].eq(False)]

    blunders_indices = weighted_sort(df_blunders.index)
    non_blunders_indices = weighted_sort(df_non_blunders.index[4:])

    ret_blunder = []
    for b_index in blunders_indices[:num_samples]:
        ret_blunder.append((boards_arr[b_index,:,:], 1, ','.join([str(v) for v in df_blunders.loc[b_index]])))

    ret_non_blunder = []
    for b_index in non_blunders_indices[:num_samples]:
        ret_non_blunder.append((boards_arr[b_index,:,:], 1, ','.join([str(v) for v in df_non_blunders.loc[b_index]])))

    return ret_blunder, ret_non_blunder

def readerWorker(inputPath, unproccessedQueue, resultsQueue, stopLoadingQueue, num_readers, testing):
    tstart = time.time()
    gamesFile = haibrid_chess_utils.LightGamesFile(inputPath, just_games = True)
    haibrid_chess_utils.printWithDate(f"Reader created", flush = True)
    try:
        tLast = time.time()
        games_bundle = []
        for i, (_, gs) in enumerate(gamesFile):
            games_bundle.append(gs)
            if len(games_bundle) >= game_per_put:
                unproccessedQueue.put(games_bundle, True)
                games_bundle = []
            if i % 100 == 0 and  time.time() - tLast > logging_delay:
                tLast = time.time()
                haibrid_chess_utils.printWithDate(f"Loaded {i} games, input queue depth: {unproccessedQueue.qsize()}, ouput queue depth: {resultsQueue.qsize()}", flush = True)
                if testing and i > 100000:
                    break
                try:
                    stopLoading = stopLoadingQueue.get_nowait()
                except queue.Empty:
                    pass
                else:
                    if stopLoading == 'kill':
                        haibrid_chess_utils.printWithDate(f"Killed by max_bs, ending after {i} games loaded")
                        break
                    else:
                        raise RuntimeError(f"{stopLoading} in wrong queue")
    except EOFError:
        pass
    if len(games_bundle) > 0:
        unproccessedQueue.put(games_bundle, True)

    haibrid_chess_utils.printWithDate(f"Done loading Queue in {humanize.naturaldelta(time.time() - tstart)}, sending kills")
    for i in range(num_readers):
        #haibrid_chess_utils.printWithDate(f"Putting kill number {i} in queue")
        unproccessedQueue.put('kill', True, 100)

def pgnParser(inputQueue, outputQueue, num_splits, min_elo, max_elo, allow_negative_loss, allow_low_time):
    while True:
        try:
            dats = inputQueue.get()
        except queue.Empty:
            break
        if dats == 'kill':
            outputQueue.put('kill', True)
            break
        else:
            for dat in dats:
                try:
                    b_vals, nb_vals = parseLines(dat, num_splits, min_elo, max_elo, allow_negative_loss, allow_low_time)
                except KeyboardInterrupt:
                    raise
                except:
                    haibrid_chess_utils.printWithDate('error:')
                    haibrid_chess_utils.printWithDate(dat)
                    raise
                if len(b_vals) > 0 or len(nb_vals) > 0:
                    outputQueue.put((b_vals, nb_vals), True)

def unload_bin_dict(bins, batch_size, myzip, blunder = False):
    count = 0
    for k in list(bins.keys()):
        a = bins[k]
        #import pdb; pdb.set_trace()
        if len(a) >= batch_size:

            x_vals = []
            y_vals = []
            dat_vals = []
            for x, y, d in a:
                x_vals.append(x)
                y_vals.append(y)
                dat_vals.append(d)
            if blunder:
                with myzip.open(f'{k:08.0f}_b_x.npy', 'w') as f:
                    #print(np.stack(x_vals)[0,0])
                    #print(dat_vals[2])
                    np.save(f, np.stack(x_vals))
                with myzip.open(f'{k:08.0f}_b_y.npy', 'w') as f:
                    np.save(f, np.stack(y_vals))
                with myzip.open(f'{k:08.0f}_b_dat.csv', 'w') as f:
                    f.write((','.join(parseHeader) + '\n' + '\n'.join(dat_vals) + '\n').encode('utf8'))
            else:
                with myzip.open(f'{k:08.0f}_nb_x.npy', 'w') as f:
                    np.save(f, np.stack(x_vals))
                with myzip.open(f'{k:08.0f}_nb_y.npy', 'w') as f:
                    np.save(f, np.stack(y_vals))
                with myzip.open(f'{k:08.0f}_nb_dat.csv', 'w') as f:
                    f.write((','.join(parseHeader) + '\n' + '\n'.join(dat_vals) + '\n').encode('utf8'))
            del bins[k]
            count += 1
    return count

def writerWorker(outputFile, inputQueue, stopLoadingQueue, num_readers, num_splits, batch_size, max_bs):
    i = -1
    num_kill_remaining = num_readers
    tstart = time.time()
    haibrid_chess_utils.printWithDate("Writer created")

    last_b_batch = 0
    last_nb_batch = 0
    num_bs = 0
    b_bins = {}
    nb_bins = {}

    if not os.path.isdir(os.path.dirname(outputFile)) and len(os.path.dirname(outputFile)) > 0:
        os.makedirs(os.path.dirname(outputFile), exist_ok=True)

    with zipfile.ZipFile(outputFile, 'w') as myzip:
        haibrid_chess_utils.printWithDate(f"Created: {outputFile}")
        tLast = time.time()
        while True:
            unload_b_bins = False
            unload_nb_bins = False
            while len(b_bins) < num_splits:
                b_bins[last_b_batch] = []
                last_b_batch += 1
            while len(nb_bins) < num_splits:
                nb_bins[last_nb_batch] = []
                last_nb_batch += 1
            try:
                dat_vals = inputQueue.get()
            except queue.Empty:
                #Should never happen
                break
            try:
                b_vals, nb_vals = dat_vals
                print("-------")
                #print(type(nb_vals[0]))
                #print(type(b_vals[0]))
                print(b_vals[0][0][0])
                print(b_vals[0][0].shape)
                print(b_vals[0][2])
                for _, a in b_bins.items():
                    try:
                        a.append(b_vals.pop())
                        if len(a) >= batch_size:
                            unload_b_bins = True
                    except IndexError:
                        break
                for _, a in nb_bins.items():
                    try:
                        a.append(nb_vals.pop())
                        if len(a) >= batch_size:
                            unload_nb_bins = True
                    except IndexError:
                        break
            except:
                if dat_vals == 'kill':
                    num_kill_remaining -= 1
                    if num_kill_remaining <= 0:
                        break
                else:
                    raise
            else:
                if unload_nb_bins:
                    unload_bin_dict(nb_bins, batch_size, myzip)
                if unload_b_bins:
                    num_bs += unload_bin_dict(b_bins, batch_size, myzip, blunder = True)
                i += 1
                if num_bs >= max_bs:
                    haibrid_chess_utils.printWithDate(f"Max bs hit stopping after {i} games")
                    stopLoadingQueue.put('kill')
                    break
                if i % 100 == 0 and  time.time() - tLast > logging_delay:
                    tLast = time.time()
                    haibrid_chess_utils.printWithDate(f"Written {i} games {last_b_batch} b batches {last_nb_batch} nb batches in {humanize.naturaldelta(time.time() - tstart)}, doing {(i + 1) /(time.time() - tstart):.0f} games a second", flush = True)

def setupProcessors(gamesPath, ouput_path, manager, worker_pool, io_pool, poolSize, num_boards, batch_size, queueSize, min_elo, max_elo, allow_negative_loss, allow_low_time, max_bs, testing):
    unproccessedQueue = manager.Queue(queueSize)
    resultsQueue = manager.Queue(queueSize)
    stopLoadingQueue = manager.Queue()

    parsers = []
    for _ in range(poolSize):
        parser = worker_pool.apply_async(pgnParser, (unproccessedQueue, resultsQueue, num_boards, min_elo, max_elo, allow_negative_loss, allow_low_time))
        parsers.append(parser)
    haibrid_chess_utils.printWithDate(f"Started {len(parsers)} parsers")

    pgnReader = io_pool.apply_async(readerWorker, (gamesPath, unproccessedQueue, resultsQueue, stopLoadingQueue, len(parsers), testing))
    haibrid_chess_utils.printWithDate(f"loader created")

    writer = io_pool.apply_async(writerWorker, (ouput_path, resultsQueue, stopLoadingQueue, len(parsers), num_boards, batch_size, max_bs))
    haibrid_chess_utils.printWithDate(f"Started writer")

    return pgnReader, parsers, writer, unproccessedQueue, resultsQueue, stopLoadingQueue

def cleanup(pgnReader, gameParsers, writer):
    while not pgnReader.ready():
        if writer.ready() and not pgnReader.ready():
            haibrid_chess_utils.printWithDate(writer.get())
        for w in gameParsers:
            if w.ready() and  not pgnReader.ready():
                haibrid_chess_utils.printWithDate(w.get())
        time.sleep(1)

    pgnReader.get()
    haibrid_chess_utils.printWithDate(f"Done reading")

    for w in gameParsers:
        w.get()
    haibrid_chess_utils.printWithDate(f"Done parsing")

    writer.get()
    haibrid_chess_utils.printWithDate(f"Done writing")


@haibrid_chess_utils.logged_main
def main():
    parser = argparse.ArgumentParser(description='Convert PGN file to zip of the binaries', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('input', help='input PGNs')
    parser.add_argument('output', help='output zip')

    parser.add_argument('--pool', type=int, help='number of simultaneous jobs procesing the file, reading and writing are two more', default = 48)
    parser.add_argument('--num_boards', type=int, help='Max number of boards of each class to take from each game', default = 5)
    parser.add_argument('--batch_size', type=int, help='Number of boards per batch', default = 100)
    parser.add_argument('--max_bs', type=int, help='Max number of blunder files to create', default = 9999999999)
    parser.add_argument('--queueSize', type=int, help='Max number of games to cache', default = 1000)

    parser.add_argument('--min_elo', type=int, help='min active elo', default = 1000)
    parser.add_argument('--max_elo', type=int, help='min active elo', default = 9999999999)
    parser.add_argument('--allow_negative_loss', type=bool, help='allow winrate losses below 0', default = False)
    parser.add_argument('--allow_low_time', type=bool, help='Include low time moves', default = False)

    parser.add_argument('--testing', help='Make run on only first 1000', default = False, action='store_true')

    args = parser.parse_args()

    if args.testing:
        haibrid_chess_utils.printWithDate(f"TESTING RUN", colour = 'red')

    haibrid_chess_utils.printWithDate(f"Starting PGN conversion of {args.input} writing to {args.output}")

    tstart = time.time()
    with multiprocessing.Manager() as manager:
        with multiprocessing.Pool(args.pool) as worker_pool, multiprocessing.Pool(3) as io_pool:
            pgnReader, gameParsers, writer, unproccessedQueue, resultsQueue, stopQueue = setupProcessors(args.input, args.output, manager, worker_pool, io_pool, args.pool, args.num_boards, args.batch_size, args.queueSize, args.min_elo, args.max_elo, args.allow_negative_loss, args.allow_low_time, args.max_bs, args.testing)

            haibrid_chess_utils.printWithDate(f"Done setting up Queues in {humanize.naturaldelta(time.time() - tstart)}, waiting for reading to finish")

            cleanup(pgnReader, gameParsers, writer)

    haibrid_chess_utils.printWithDate(f"Done everything in {humanize.naturaldelta(time.time() - tstart)}, exiting")

if __name__ == '__main__':
    main()
