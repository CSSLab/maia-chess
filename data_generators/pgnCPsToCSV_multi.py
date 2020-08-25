#Working functions used by single

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
import traceback

import chess
import chess.pgn

import haibrid_chess_utils

logging_delay = 30 # in seconds

num_move_per_game = 35 #For clock remaing calculations
low_time_threshold = 30
winrate_blunder_threshold = .1

time_regex = re.compile(r'\[%clk (\d+):(\d+):(\d+)\]')
eval_regex = re.compile(r'\[%eval ([0-9.+-]+)\]')

properties = [
    'game_id',
    'type',
    'result',
    'white_player',
    'black_player',
    'white_elo',
    'black_elo',
    'time_control',
    'termination',
    'move_ply',
    'move',
    'cp',
    'cp_rel',
    'cp_loss',
    'is_blunder',
    'winrate',
    'winrate_loss',
    'blunder_wr',
    'opp_winrate',
    'white_active',
    'active_elo',
    'opponent_elo',
    'active_won',
    'no_winner',
    'clock',
    'opp_clock',
    'clock_percent',
    'opp_clock_percent',
    'low_time',
    'board',
]

properties += haibrid_chess_utils.board_stats_header

def readerWorker(inputPath, unproccessedQueue, resultsQueue, name, num_readers):
    tstart = time.time()
    gamesFile = haibrid_chess_utils.LightGamesFile(inputPath, just_games = True)
    try:
        tLast = time.time()
        for i, (_, gs) in enumerate(gamesFile):
            unproccessedQueue.put(gs, True, 1000)
            if i % 1000 == 0 and  time.time() - tLast > logging_delay:
                tLast = time.time()
                haibrid_chess_utils.printWithDate(f"{name} Loaded {i} games, input queue depth: {unproccessedQueue.qsize()}, ouput queue depth: {resultsQueue.qsize()}", flush = True)
    except EOFError:
        pass

    haibrid_chess_utils.printWithDate(f"{name} Done loading Queue in {humanize.naturaldelta(time.time() - tstart)}, sending kills")
    for i in range(num_readers):
        #haibrid_chess_utils.printWithDate(f"Putting kill number {i} in queue")
        unproccessedQueue.put('kill', True, 100)


def writerWorker(outputFile, inputQueue, num_readers, name):
    i = -1
    num_kill_remaining = num_readers
    tstart = time.time()
    haibrid_chess_utils.printWithDate("Writer created")
    with bz2.open(outputFile, 'wb') as f:
        haibrid_chess_utils.printWithDate(f"Created: {outputFile}")
        f.write((','.join(haibrid_chess_utils.full_csv_header) + '\n').encode('utf8'))
        tLast = time.time()
        while True:
            try:
                dat = inputQueue.get()
            except queue.Empty:
                #Should never happen
                break
            try:
                f.write(dat)
            except TypeError:
                if dat == 'kill':
                    num_kill_remaining -= 1
                    if num_kill_remaining <= 0:
                        break
                else:
                    raise
            else:
                i += 1
                if i % 1000 == 0 and  time.time() - tLast > logging_delay:
                    tLast = time.time()
                    haibrid_chess_utils.printWithDate(f"{name} Written {i} games in {humanize.naturaldelta(time.time() - tstart)}, doing {(i + 1) /(time.time() - tstart):.0f} games a second", flush = True)
    haibrid_chess_utils.printWithDate("Received shutdown signal to writer")
    haibrid_chess_utils.printWithDate(f"Done a total of {i} games in {humanize.naturaldelta(time.time() - tstart)}")

def gamesConverter(inputQueue, outputQueue):
    #haibrid_chess_utils.printWithDate("Converter created")
    while True:
        try:
            #print('qsize', inputQueue.qsize())
            dat = inputQueue.get()
        except queue.Empty:
            break
        if dat == 'kill':
            outputQueue.put('kill', True, 1000)
            break
        else:
            try:
                s = haibrid_chess_utils.gameToCSVlines(dat)
            except haibrid_chess_utils.NoStockfishEvals:
                pass
            except:
                haibrid_chess_utils.printWithDate('error:')
                haibrid_chess_utils.printWithDate(dat)
                haibrid_chess_utils.printWithDate(traceback.format_exc())
                raise
            else:
                if len(s) > 0:
                    lines = '\n'.join(s) + '\n'
                    outputQueue.put(lines.encode('utf8'), True, 1000)
    haibrid_chess_utils.printWithDate("Received shutdown signal to Converter", flush = True)

def makeCSVlines(gameStr):
    """Depricated, now we're using haibrid_chess_utils.gameToCSVlines"""

    game = chess.pgn.read_game(io.StringIO(gameStr))

    gameVals = []
    retVals = []

    #game_id
    gameVals.append(game.headers['Site'].split('/')[-1])

    #type
    gameVals.append(game.headers['Event'].split(' tournament')[0].replace(' game', '').replace('Rated ', ''))

    #result
    gameVals.append(game.headers['Result'])

    #white_player
    gameVals.append(game.headers['White'])

    #black_player
    gameVals.append(game.headers['Black'])

    #white_elo
    gameVals.append(game.headers['WhiteElo'])

    #black_elo
    gameVals.append(game.headers['BlackElo'])

    #time_control
    gameVals.append(game.headers['TimeControl'])

    #termination
    gameVals.append(game.headers['Termination'])

    white_won = game.headers['Result'] == '1-0'
    no_winner = game.headers['Result'] not in  ['1-0', '0-1']

    time_per_player = haibrid_chess_utils.time_control_to_secs(game.headers['TimeControl'], moves_per_game = num_move_per_game)

    board = game.board()
    cp_board = .1
    cp_str_last = '0.1'
    cp_rel_str_last = '0.1'
    no_time = False
    last_clock_seconds = -1
    for i, node in enumerate(game.mainline()):
        moveVals = []
        fen = str(board.fen())
        is_white = fen.split(' ')[1] == 'w'
        try:
            cp_str = eval_regex.search(node.comment).group(1)
        except AttributeError:
            break
        else:
            try:
                cp_after = float(cp_str)
            except ValueError:
                if '-' in node.comment:
                    cp_after = float('-inf')
                else:
                    cp_after = float('inf')
            if not is_white:
                cp_after *= -1
            cp_rel_str = str(cp_after)

        if not no_time:
            try:
                timesRe = time_regex.search(node.comment)

                clock_seconds = int(timesRe.group(1)) * 60 * 60 + int(timesRe.group(2)) * 60  + int(timesRe.group(3))

            except AttributeError:
                no_time = True
                clock_seconds = time_per_player

            # make equal on first move
            if last_clock_seconds < 0:
                last_clock_seconds = clock_seconds

        act_elo = game.headers['WhiteElo'] if is_white else game.headers['BlackElo']
        opp_elo = game.headers['BlackElo'] if is_white else game.headers['WhiteElo']
        if no_winner:
            act_won = False
        elif is_white:
            act_won = white_won
        else:
            act_won = not white_won

        cp_loss = cp_board - cp_after # CPs are all relative

        winrate_current = haibrid_chess_utils.cp_to_winrate(cp_board, elo = act_elo)

        winrate_loss = winrate_current - haibrid_chess_utils.cp_to_winrate(cp_after, elo = act_elo)

        winrate_opp = haibrid_chess_utils.cp_to_winrate(-cp_board, elo = opp_elo)

        #This might need to be cleaned up soon
        #If only there was a way to map strings to stuff ...

        #move_ply
        moveVals.append(str(i + 1))
        #move
        moveVals.append(str(node.move))
        #cp
        moveVals.append(str(cp_str_last))
        #cp_rel
        moveVals.append(str(cp_rel_str_last))
        #cp_loss
        moveVals.append(f"{cp_loss:.2f}")
        #is_blunder
        moveVals.append(str(cp_loss >= 2))
        #winrate
        moveVals.append(f"{winrate_current:.4f}")
        #winrate_loss
        moveVals.append(f"{winrate_loss:.4f}")
        #blunder_wr
        moveVals.append(str(winrate_loss > winrate_blunder_threshold))
        #opp_winrate
        moveVals.append(f"{winrate_opp:.4f}")
        #white_active
        moveVals.append(str(is_white))
        #active_elo
        moveVals.append(str(act_elo))
        #opponent_elo
        moveVals.append(str(opp_elo))
        #active_won
        moveVals.append(str(act_won))
        #no_winner
        moveVals.append(str(no_winner))
        #clock
        moveVals.append(str(clock_seconds))
        #opp_clock
        moveVals.append(str(last_clock_seconds))
        #clock_percent
        moveVals.append(f"{1 - clock_seconds/time_per_player:.3f}")
        #opp_clock_percent
        moveVals.append(f"{1 - last_clock_seconds/time_per_player:.3f}")
        #low_time
        moveVals.append(str(bool(clock_seconds < low_time_threshold)))

        #board
        moveVals.append(fen)

        moveVals += [str(v) for k,v in sorted(haibrid_chess_utils.board_stats(fen).items(), key = lambda x : x[0])]

        board.push(node.move)

        retVals.append(','.join(gameVals + moveVals))
        cp_board = -1 * cp_after
        cp_str_last = cp_str
        cp_rel_str_last = cp_rel_str
        last_clock_seconds = clock_seconds

    return retVals

def processPGN(gamesPath, inputName, outputName, queueSize, poolSize, manager, workers_pool, io_pool):
    unproccessedQueue = manager.Queue(queueSize)
    resultsQueue = manager.Queue(queueSize)

    readers = []
    for _ in range(poolSize - 1):
        reader = workers_pool.apply_async(gamesConverter, (unproccessedQueue, resultsQueue))
        readers.append(reader)
    haibrid_chess_utils.printWithDate(f"{inputName} Started {len(readers)} readers", flush = True)

    pgnReader = io_pool.apply_async(readerWorker, (gamesPath, unproccessedQueue, resultsQueue, inputName, len(readers)))
    haibrid_chess_utils.printWithDate(f"{inputName} loader created")

    writer = io_pool.apply_async(writerWorker, (outputName, resultsQueue, len(readers), inputName))
    haibrid_chess_utils.printWithDate(f"{inputName} Started writer for: {inputName}", flush = True)

    return pgnReader, readers, writer, unproccessedQueue, resultsQueue

def cleanup(pgnReaders, gameReaders, writers):
    while len(pgnReaders) > 0:
        for i in list(pgnReaders.keys()):
            if pgnReaders[i].ready():
                haibrid_chess_utils.printWithDate(f"{i} Done reading")
                pgnReaders[i].get()
                del pgnReaders[i]
        time.sleep(10)

    for i in list(gameReaders.keys()):
        for w in gameReaders[i]:
            w.get()
        haibrid_chess_utils.printWithDate(f"{i} Done processing")

    for i in list(writers.keys()):
        writers[i].get()

@haibrid_chess_utils.logged_main
def main():
    parser = argparse.ArgumentParser(description='process PGN file with stockfish annotaions into a csv file', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('inputs', nargs = '+', help='input PGNs')
    parser.add_argument('outputDir', help='output CSVs dir')

    parser.add_argument('--pool', type=int, help='number of simultaneous jobs running per fil', default = 20)
    #parser.add_argument('--readers', type=int, help='number of simultaneous reader running per inputfile', default = 24)
    parser.add_argument('--queueSize', type=int, help='Max number of games to cache', default = 1000)

    args = parser.parse_args()

    haibrid_chess_utils.printWithDate(f"Starting CSV conversion of {', '.join(args.inputs)} writing to {args.outputDir}")

    os.makedirs(args.outputDir, exist_ok=True)

    names = {}
    for n in args.inputs:
        name = os.path.basename(n).split('.')[0]
        outputName = os.path.join(args.outputDir, f"{name}.csv.bz2")
        names[n] = (name, outputName)


        haibrid_chess_utils.printWithDate(f"Loading file: {name}")
    haibrid_chess_utils.printWithDate(f"Starting main loop")

    tstart = time.time()

    pgnReaders = {}
    gameReaders = {}
    writers = {}
    queues = {}

    with multiprocessing.Manager() as manager:
        with multiprocessing.Pool(args.pool * len(names)) as workers_pool, multiprocessing.Pool(len(names) * 2 + 4) as io_pool:
            for p, (i, o) in names.items():
                pgnReader, gameReader, writer, unproccessedQueue, resultsQueue = processPGN(p, i, o, args.queueSize, args.pool, manager, workers_pool, io_pool)
                pgnReaders[i] = pgnReader
                gameReaders[i] = gameReader
                writers[i] = writer
                queues[i] = (unproccessedQueue, resultsQueue)

            haibrid_chess_utils.printWithDate(f"Done loading Queues in {humanize.naturaldelta(time.time() - tstart)}, waiting for reading to finish")

            cleanup(pgnReaders, gameReaders, writers)

    haibrid_chess_utils.printWithDate(f"Done everything in {humanize.naturaldelta(time.time() - tstart)}, exiting")


if __name__ == '__main__':
    main()
