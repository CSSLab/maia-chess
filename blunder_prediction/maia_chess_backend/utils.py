import chess
import chess.pgn
import re
import datetime
import json
import os.path
import io
import bz2
import multiprocessing
import functools

import pandas
import pytz
tz = pytz.timezone('Canada/Eastern')

low_time_threshold = 30
winrate_blunder_threshold = .1

time_regex = re.compile(r'\[%clk (\d+):(\d+):(\d+)\]')
eval_regex = re.compile(r'\[%eval ([0-9.+-]+)|(#(-)?[0-9]+)\]')
low_time_re = re.compile(r'(\d+\. )?\S+ \{ \[%clk 0:00:[210]\d\]')

class NoStockfishEvals(Exception):
    pass

pieces = {
    'pawn' : "P",
    'knight' : "N",
    'bishop' : "B",
    'rook' : "R",
    'queen' : "Q",
    #'king' : "K", #Ignoring kings for the counts
}

board_stats_header = [
        'active_bishop_count',
        'active_knight_count',
        'active_pawn_count',
        'active_queen_count',
        'active_rook_count',
        'is_check',
        'num_legal_moves',
        'opp_bishop_count',
        'opp_knight_count',
        'opp_pawn_count',
        'opp_queen_count',
        'opp_rook_count',
]

moveRe = re.compile(r"^\S+")
probRe = re.compile(r"\(P: +([^)%]+)%\)")
uRe = re.compile(r"\(U: +([^)]+)\)")
qRe = re.compile(r"\(Q: +([^)]+)\)")
nRe = re.compile(r" N: +(\d+) \(")

fenComps = 'rrqn2k1/8/pPp4p/2Pp1pp1/3Pp3/4P1P1/R2NB1PP/1Q4K1 w KQkq - 0 1'.split()

cpLookup = None
cpLookup_simple = None

def profile_helper(target):
    try:
        return profile(target)
    except NameError:
        @functools.wraps(target)
        def no_op_wrap(*args, **kwds):
            return target(*args, **kwds)
        return no_op_wrap

def board_to_lichess(b_str):
    return f"https://lichess.org/analysis/standard/{str(b_str).replace(' ', '_')}"

def load_partial_bz(path):
    lines = []
    try:
        with bz2.open(path, 'rt') as f:
            for line in f:
                lines.append(line)
    except EOFError:
        pass
    return io.StringIO(''.join(lines))

def remove_low_time(g_str):
    r = low_time_re.search(g_str)
    if r is None:
        return g_str
    end = g_str[-20:].split(' ')[-1]
    return g_str[:r.span()[0]] + end

def cp_to_winrate(cp, lookup_file = os.path.join(os.path.dirname(__file__), '../data/cp_winrate_lookup_simple.json'), allow_nan = False):
    global cpLookup_simple
    try:
        cp = int(float(cp) * 10) / 10
    except OverflowError:
        return float("nan")
    except ValueError:
        #This can be caused by a bunch of other things too so this option is dangerous
        if allow_nan:
            return float("nan")
        else:
            raise
    if cpLookup_simple is None:
        with open(lookup_file) as f:
            cpLookup_str = json.load(f)
            cpLookup_simple = {float(k) : wr for k, wr in cpLookup_str.items()}
    try:
        return cpLookup_simple[cp]
    except KeyError:
        return float("nan")

def cp_to_winrate_elo(cp, elo = 1500, lookup_file = os.path.join(os.path.dirname(__file__), '../data/cp_winrate_lookup.json'), allow_nan = False):
    global cpLookup
    try:
        cp = int(float(cp) * 10) / 10
        elo = int(float(elo)//100) * 100
    except OverflowError:
        return float("nan")
    except ValueError:
        #This can be caused by a bunch of other things too so this option is dangerous
        if allow_nan:
            return float("nan")
        else:
            raise
    if cpLookup is None:
        with open(lookup_file) as f:
            cpLookup_str = json.load(f)
            cpLookup = {}
            for k, v in cpLookup_str.items():
                cpLookup[int(k)] = {float(k) : wr for k, wr in v.items()}
    try:
        return cpLookup[elo][cp]
    except KeyError:
        return float("nan")

def board_stats(input_board, board_fen = None):
    if isinstance(input_board, str):
        board = chess.Board(fen=input_board)
        board_fen = input_board
    else:
        board = input_board
        if board_fen is None:
            board_fen = input_board.fen()
    board_str = board_fen.split(' ')[0]
    dat = {
        'num_legal_moves' : len(list(board.legal_moves)),
        'is_check' : int(board.is_check())
    }
    for name, p in pieces.items():
        if active_is_white(board_fen):
            dat[f'active_{name}_count'] = board_fen.count(p)
            dat[f'opp_{name}_count'] = board_fen.count(p.lower())
        else:
            dat[f'active_{name}_count'] = board_fen.count(p.lower())
            dat[f'opp_{name}_count'] = board_fen.count(p)
    return dat

def active_is_white(fen_str):
    return fen_str.split(' ')[1] == 'w'

def time_control_to_secs(timeStr, moves_per_game = 35):
    if timeStr == '-':
        return 10800 # 180 minutes per side max on lichess
    else:
        t_base, t_add = timeStr.split('+')
        return int(t_base) + int(t_add) * moves_per_game

def fen_extend(s):
    splitS = s.split()
    return ' '.join(splitS + fenComps[len(splitS):])

def fen(s):
    return chess.Board(fen_extend(s))

def gameToFenSeq(game):
    headers = dict(game)
    moves = getBoardMoveMap(game)
    return {'headers' : headers, 'moves' : moves}

def getMoveStats(s):
    return {
        'move' : moveRe.match(s).group(0),
        'prob' : float(probRe.search(s).group(1)) / 100,
        'U' : float(uRe.search(s).group(1)),
        'Q' : float(qRe.search(s).group(1)),
        'N' : float(nRe.search(s).group(1)),
    }

def movesToUCI(moves, board):
    if isinstance(board, str):
        board = fen(board)
    moveMap = {}
    for m in moves:
        board.push_san(m)
        moveMap[m] = board.pop().uci()
    return moveMap

def getSeqs(inputNode):
    retSeqs = []
    for k, v in list(inputNode.items()):
        if k == 'hits' or k == 'sfeval':
            pass
        elif len(v) <= 2:
            retSeqs.append([k])
        else:
            retSeqs +=  [[k] + s for s in getSeqs(v)]
    return retSeqs

def moveSeqToBoard(seq):
    board = chess.Board()
    for m in seq:
        board.push_san(m.replace('?', '').replace('!', ''))
    return board

def makeFEN(seq):
    board = moveSeqToBoard(seq)
    return ','.join(seq), board.fen(), len(list(board.legal_moves))

def moveTreeLookup(d, procs = 64):
    sequences = getSeqs(d)
    with multiprocessing.Pool(procs) as pool:
        maps = pool.map(makeFEN, sequences)
    return maps

colours = {
    'blue' : '\033[94m',
    'green' : '\033[92m',
    'yellow' : '\033[93m',
    'red' : '\033[91m',
    'pink' : '\033[95m',
}
endColour = '\033[0m'


def printWithDate(s, colour = None, **kwargs):
    if colour is None:
        print(f"{datetime.datetime.now(tz).strftime('%Y-%m-%d %H:%M:%S')} {s}", **kwargs)
    else:
        print(f"{datetime.datetime.now(tz).strftime('%Y-%m-%d %H:%M:%S')}{colours[colour]} {s}{endColour}", **kwargs)

all_per_game_vals = [
    'game_id',
    'type',
    'result',
    'white_player',
    'black_player',
    'white_elo',
    'black_elo',
    'time_control',
    'num_ply',
    'termination',
    'white_won',
    'black_won',
    'no_winner',
]


per_game_funcs = {
    'game_id' : lambda x : x['Site'].split('/')[-1],
    'type' : lambda x : x['Event'].split(' tournament')[0].replace(' game', '').replace('Rated ', ''),
    'result' : lambda x : x['Result'],
    'white_player' : lambda x : x['White'],
    'black_player' : lambda x : x['Black'],
    'white_elo' : lambda x : x['WhiteElo'],
    'black_elo' : lambda x : x['BlackElo'],
    'time_control' : lambda x : x['TimeControl'],
    'termination' : lambda x : x['Termination'],
    'white_won' : lambda x : x['Result'] == '1-0',
    'black_won' : lambda x : x['Result'] == '0-1',
    'no_winner' : lambda x : x['Result'] not in  ['1-0', '0-1'],
}

all_per_move_vals = [
    'move_ply',
    'move',
    'cp',
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
    'is_capture',
    'clock',
    'opp_clock',
    'clock_percent',
    'opp_clock_percent',
    'low_time',
    'board',
]

per_move_funcs = {
    'move_ply' : lambda x : x['i'],
    'move' : lambda x : x['node'].move,
    'cp' : lambda x : x['cp_str_last'],
    'cp_rel' : lambda x : x['cp_rel_str_last'],
    'cp_loss' : lambda x : f"{x['cp_loss']:.2f}",
    'is_blunder_cp' : lambda x : x['cp_loss'] >= 2,
    'winrate' : lambda x : f"{x['winrate_current']:.4f}",
    'winrate_elo' : lambda x : f"{x['winrate_current_elo']:.4f}",
    'winrate_loss' : lambda x :
    f"{x['winrate_loss']:.4f}",
    'is_blunder_wr' : lambda x : x['winrate_loss'] > winrate_blunder_threshold,
    'opp_winrate' : lambda x : f"{x['winrate_opp']:.4f}",
    'white_active' : lambda x : x['is_white'],
    'active_elo' : lambda x : x['act_elo'],
    'opponent_elo' : lambda x : x['opp_elo'],
    'active_won' : lambda x : x['act_won'],
    'is_capture' : lambda x : x['board'].is_capture(x['node'].move),
    'clock' : lambda x : x['clock_seconds'],
    'opp_clock' : lambda x : x['last_clock_seconds'],
    'clock_percent' : lambda x : '' if x['no_time'] else f"{1 - x['clock_seconds']/x['time_per_player']:.3f}",
    'opp_clock_percent' : lambda x : '' if x['no_time'] else f"{1 - x['last_clock_seconds']/x['time_per_player']:.3f}",
    'low_time' : lambda x : '' if x['no_time'] else x['clock_seconds'] < low_time_threshold,
    'board' : lambda x : x['fen'],
}

full_csv_header = all_per_game_vals + all_per_move_vals + board_stats_header

def gameToDF(input_game, per_game_vals = None, per_move_vals = None, with_board_stats = True, allow_non_sf = False):
    """Hack to make dataframes instead of CSVs while maintaining the smae code as much as possible"""

    csv_lines = gameToCSVlines(input_game, per_game_vals = per_game_vals, per_move_vals = per_move_vals, with_board_stats = with_board_stats, allow_non_sf = allow_non_sf)

    csv_header = list(per_game_vals) + list(per_move_vals)
    if with_board_stats:
        csv_header = csv_header + board_stats_header

    # a hack, but makes things consistant
    return pandas.read_csv(io.StringIO('\n'.join(csv_lines)), names = csv_header)

def gameToCSVlines(input_game, per_game_vals = None, per_move_vals = None, with_board_stats = True, allow_non_sf = False):
    """Main function in created the datasets

    There's per game and per board stuff that needs to be calculated, with_board_stats is just a bunch of material counts.

    The different functions that are applied are simple and mostly stored in two dicts: per_game_funcs and per_move_funcs. per_move_funcs are more complicated and can depend on a bunch of stuff so they just get locals() as an input which is a hack, but it works. They all used to be in the local namespace this was just much simpler than rewriting all of them.
    """
    #defaults to everything
    if isinstance(input_game, str):
        game = chess.pgn.read_game(io.StringIO(input_game))
    else:
        game = input_game

    if per_game_vals is None:
        per_game_vals = all_per_game_vals
    if per_move_vals is None:
        per_move_vals = all_per_move_vals

    gameVals = []
    retVals = []

    for n in per_game_vals:
        try:
            gameVals.append(per_game_funcs[n](game.headers))
        except KeyError:
            if n == 'num_ply':
                gameVals.append(len(list(game.mainline())))
            else:
                raise

    gameVals = [str(v) for v in gameVals]

    white_won = game.headers['Result'] == '1-0'
    no_winner = game.headers['Result'] not in  ['1-0', '0-1']

    time_per_player = time_control_to_secs(game.headers['TimeControl'])

    board = game.board()
    cp_board = .1
    cp_str_last = '0.1'
    cp_rel_str_last = '0.1'
    no_time = False
    last_clock_seconds = -1

    for i, node in enumerate(game.mainline()):
        comment = node.comment.replace('\n', ' ')
        moveVals = []
        fen = str(board.fen())
        is_white = fen.split(' ')[1] == 'w'

        try:
            cp_re = eval_regex.search(comment)
            cp_str = cp_re.group(1)
        except AttributeError:
            if i > 2:
                #Indicates mate
                if not is_white:
                    cp_str = '#-0'
                    cp_after = float('-inf')
                else:
                    cp_str = '#0'
                    cp_after = float('inf')
            else:
                if not allow_non_sf:
                    break
                else:
                    cp_str = 'nan'
                    cp_after = float('nan')
                #raise AttributeError(f"weird comment found: {node.comment}")
        else:
            if cp_str is not None:
                try:
                    cp_after = float(cp_str)
                except ValueError:
                    if '-' in comment:
                        cp_after = float('-inf')
                    else:
                        cp_after = float('inf')
            else:
                if cp_re.group(3) is None:
                    cp_after = float('inf')
                else:
                    cp_after = float('-inf')
        if not is_white:
            cp_after *= -1
        cp_rel_str = str(-cp_after)
        if not no_time:
            try:
                timesRe = time_regex.search(comment)

                clock_seconds = int(timesRe.group(1)) * 60 * 60 + int(timesRe.group(2)) * 60  + int(timesRe.group(3))

            except AttributeError:
                no_time = True
                clock_seconds = ''

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

        winrate_current_elo = cp_to_winrate_elo(cp_board, elo = act_elo, allow_nan = allow_non_sf)
        winrate_current = cp_to_winrate(cp_board, allow_nan = allow_non_sf)

        winrate_loss = winrate_current -cp_to_winrate(cp_after, allow_nan = allow_non_sf)

        winrate_opp = cp_to_winrate(-cp_board, allow_nan = allow_non_sf)

        for n in per_move_vals:
            moveVals.append(per_move_funcs[n](locals()))

        if with_board_stats:
            moveVals += [str(v) for k,v in sorted(board_stats(board, fen).items(), key = lambda x : x[0])]

        board.push(node.move)

        moveVals = [str(v) for v in moveVals]

        retVals.append(','.join(gameVals + moveVals))
        cp_board = -1 * cp_after
        cp_str_last = cp_str
        cp_rel_str_last = cp_rel_str
        last_clock_seconds = clock_seconds
    if len(retVals) < 1:
        raise NoStockfishEvals("No evals found in game")
    return retVals
