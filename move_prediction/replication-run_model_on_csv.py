import argparse
import os
import os.path
import bz2
import csv
import multiprocessing
import humanize
import time
import queue

import chess

import maia_chess_backend

@maia_chess_backend.logged_main
def main():
    parser = argparse.ArgumentParser(description='Run model on all the lines of the csv', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('nets', help='nets dir')
    parser.add_argument('input', help='input CSV')
    parser.add_argument('output', help='output dir')
    parser.add_argument('--nrows', type=int, help='number of rows to read in', default = None)
    parser.add_argument('--lc0_depth', type=int, help='Number of rollouts/nodes for lc0 engines', default = None)
    parser.add_argument('--pool_size', type=int, help='Number of models to run in parallel', default = 64)
    parser.add_argument('--queueSize', type=int, help='Max number of games to cache', default = 1000)
    parser.add_argument('--multipv', type=int, help='Number of potetial boards to consider', default = None)
    parser.add_argument('--no_hist', help='Disable history', action = 'store_true')
    args = parser.parse_args()

    maia_chess_backend.printWithDate(f"Starting model {args.nets} analysis of {args.input} writing to {args.output}")

    models = []
    for name, _, files in os.walk(args.nets):
        if 'config.yaml' in files:
            models.append(name)

    maia_chess_backend.printWithDate(f"Found {len(models)} models")
    os.makedirs(args.output, exist_ok = True)
    with multiprocessing.Pool(args.pool_size + 1) as pool, multiprocessing.Manager() as manager:
        for model_path in models:
            runModel(model_path, manager, pool, args.pool_size,args.queueSize, args.input, args.output, args.nrows, args.lc0_depth, args.multipv, args.no_hist)
    maia_chess_backend.printWithDate(f"Done all models")

def runModel(model_path, manager, pool, pool_size, queueSize, data_path, output_dir, nrows, lc0_depth, multipv, no_hist):
    unproccessedQueue = manager.Queue(queueSize)
    resultsQueue = manager.Queue(queueSize)

    try:
        model, config = maia_chess_backend.load_model_config(model_path, lc0_depth = lc0_depth)
        name = config['name']
        display_name = config['display_name']
    except NotImplementedError:
        maia_chess_backend.printWithDate(f"{model_path} model not implemented", colour = 'red')
        return
    output_file = os.path.join(output_dir, f"{name}_results.csv.bz2")
    runners = []
    for _ in range(pool_size - 1):
        runner = pool.apply_async(sequenceRunner, (unproccessedQueue, resultsQueue, model_path, lc0_depth, multipv))
        runners.append(runner)
    maia_chess_backend.printWithDate(f"{display_name} Started {len(runners)} runners", flush = True)
    writer = pool.apply_async(writerWorker, (output_file, resultsQueue, pool_size - 1, name, display_name, lc0_depth, multipv))
    tstart = time.time()
    with bz2.open(data_path, 'rt') as fin:
        reader = csv.DictReader(fin)
        board = chess.Board()
        current_game = None
        for i, row in enumerate(reader):
            if nrows is not None and i >= nrows:
                break
            if no_hist or row['game_id'] != current_game:
                current_game = row['game_id']
                board = chess.Board(fen = row['board'])
            unproccessedQueue.put((board, {
                            'game_id' : row['game_id'],
                            'move_ply' : row['move_ply'],
                            'move' : row['move'],
                            }
                        ))
            try:
                board.push_uci(row['move'])
            except ValueError:
                current_game = row['game_id']
            if i % 1000 == 0:
                maia_chess_backend.printWithDate(f"{name} row {i} in {humanize.naturaldelta(time.time() - tstart)} {unproccessedQueue.qsize()} in {resultsQueue.qsize()} out".ljust(50), end = '\r')
    for _ in range(pool_size - 1):
        unproccessedQueue.put('kill')
    writer_done = False
    if not writer_done and writer.ready():
        writer.get()
        writer_done = True
    while len(runners) > 0:
        for i in range(len(runners)):
            try:
                runners[i].get(1)
            except multiprocessing.TimeoutError:
                pass
            else:
                del runners[i]
                break
        if not writer_done and writer.ready():
            writer.get()
            writer_done = True
    if not writer_done:
        writer.get()

    maia_chess_backend.printWithDate(f"{name} done {i} rows in {humanize.naturaldelta(time.time() - tstart)}".ljust(70))

def sequenceRunner(inputQueue, outputQueue, model_path, lc0_depth, multipv):
    model, config = maia_chess_backend.load_model_config(model_path, lc0_depth = lc0_depth)
    while True:
        try:
            dat = inputQueue.get()
        except queue.Empty:
            break
        if dat == 'kill':
            outputQueue.put('kill', True)
            break
        else:
            board, row = dat
            if multipv is None:
                m_move_obj, m_cp = model.getMoveWithCP(board)
                m_move = m_move_obj.uci()
                outputQueue.put((m_move, m_cp, row))
            else:
                moves = model.getTopMovesCP(board, multipv)
                outputQueue.put(([m for m, c in moves], [c for m, c in moves], row))

def writerWorker(outputFile, inputQueue, num_readers, name, display_name, rl_depth, multipv):
    num_kill_remaining = num_readers
    with bz2.open(outputFile, 'wt') as f:
        if multipv is None:
            writer = csv.DictWriter(f, ['game_id', 'move_ply', 'player_move', 'model_move', 'model_cp', 'model_correct', 'model_name', 'model_display_name', 'rl_depth'])
        else:
            moveheader = []
            for i in range(multipv):
                moveheader+= [f'model_move_{i}', f'model_cp_{i}']
            writer = csv.DictWriter(f, ['game_id', 'move_ply', 'player_move', 'model_correct', 'model_name', 'model_display_name', 'rl_depth'] + moveheader)
        writer.writeheader()
        while True:
            dat = inputQueue.get()
            if dat == 'kill':
                num_kill_remaining -= 1
                if num_kill_remaining <= 0:
                    break
            else:
                m_move, m_cp, row = dat
                write_dict = {
                    'game_id' : row['game_id'],
                    'move_ply' : row['move_ply'],
                    'player_move' : row['move'],
                    'model_correct' : row['move'] == m_move,
                    'model_name' : name,
                    'model_display_name' : display_name,
                    'rl_depth' : rl_depth,
                }
                if multipv is None:
                    write_dict['model_move'] = m_move
                    write_dict['model_cp'] = m_cp
                else:
                    for i, m in enumerate(m_move):
                        write_dict[f'model_move_{i}'] = m
                    for i, c in enumerate(m_cp):
                        write_dict[f'model_cp_{i}'] = c
                writer.writerow(write_dict)

if __name__ == '__main__':
    main()
