import chess
import chess.engine
import chess.pgn

import pytz

import subprocess
import random
import json
import os
import os.path
import time
import datetime
import concurrent.futures

import numpy as np

from .models_loader import Trained_Model
from .utils import printWithDate

tz = pytz.timezone('Canada/Eastern')
if os.path.isfile('/u/reidmcy/.local/bin/stockfish'):
    _stockfishPath = '/u/reidmcy/.local/bin/stockfish'
else:
    _stockfishPath = '/home/reidmcy/.local/bin/stockfish'
_lc0Path = 'lc0'

_movetime = 10000

networksDir = '/u/reidmcy/chess/imitation-chess/networks'

stockfish_SKILL = [0, 3, 6, 10, 14, 16, 18, 20]
stockfish_MOVETIMES = [50, 100, 150, 200, 300, 400, 500, 1000]
stockfish_DEPTHS = [1, 1, 2, 3, 5, 8, 13, 22]

def cpToInt(cpVal):
    if cpVal.is_mate():
        if cpVal.relative.mate() < 0:
            return -1000
        else:
            return 1000
    else:
        return int(cpVal.relative.cp)

class TourneyEngine(object):
    def __init__(self, engine, name, movetime = None, nodes = None, depth = None):
        self.engine = engine
        self.name = f"{type(self).__name__} {name}"
        self.movetime = movetime
        self.depth = depth
        self.nodes = nodes

        self.limits = chess.engine.Limit(
            time = movetime,
            depth = depth,
            nodes = nodes,
        )

    def __repr__(self):
        return f"<{self.name}>"

    def __str__(self):
        return self.name

    def getTopMovesCP(self, board, num_moves):
        results = self.engine.analyse(
                board,
                self.limits,
                info = chess.engine.INFO_ALL,
                multipv = num_moves,
                )
        ret_dat = []
        for m_dict in results:
            try:
                cp = cpToInt(m_dict['score'])
            except KeyError:
                cp = 0
            ret_dat.append((m_dict['pv'][0].uci(), cp))
        return ret_dat

    def getMoveWithCP(self, board):
        result = self.getResults(board)
        try:
            cp = cpToInt(result.info['score'])
        except KeyError:
            cp = 0
        return result.move, cp

    def getMove(self, board):
        result = self.getResults(board)
        return result.move

    def getResults(self, board):
        return self.engine.play(board, self.limits, game = board, info = chess.engine.INFO_ALL)

    def getBoardChildren(self, board):
        moves_ret = {}
        for m in board.legal_moves:
            b_m = board.copy()
            b_m.push(m)
            r = self.engine.analyse(b_m, limit=self.limits, info = chess.engine.INFO_ALL, multipv = None)
            moves_ret[str(m)] = r
        return moves_ret

    def getMeanEval(self, board, depth):
        scores = []
        if depth <= 0:
            cVals = self.getBoardChildren(board)
            for m, d in cVals.items():
                scores.append(cpToInt(d['score']))
        elif depth % 2 == 1:
            b_m = board.copy()
            m = self.getMove(b_m)
            b_m.push(m)
            return self.getMeanEval(b_m, depth - 1)
        else:
            for m in board.legal_moves:
                b_m = board.copy()
                b_m.push(m)
                scores.append(self.getMeanEval(b_m, depth - 1))

        return np.mean(scores)

    def depthMovesSearch(self, board, depth = 2):
        moves = {}
        for m in sorted(board.legal_moves):
            b_m = board.copy()
            b_m.push(m)
            moves[str(m)] = self.getMeanEval(b_m, depth - 1)
        return max(moves.items(), key = lambda x : x[1])

    def __del__(self):
        try:
            try:
                self.engine.quit()
            except (chess.engine.EngineTerminatedError, concurrent.futures._base.TimeoutError):
                pass
        except AttributeError:
            pass

class _MoveHolder(object):
    def __init__(self, move):
        self.bestmove = move
        self.move = move

class _Random_Results(object):
    def __init__(self, move):
        self.move = move
        self.info = {}

class _RandomEngineBackend(object):
    def __init__(self):
        self.nextMove = None

    def position(self, board):
        self.nextMove = random.choice(list(board.legal_moves))

    def go(self, *args, **kwargs):
        return _MoveHolder(self.nextMove)

    def play(self, board, *args, **kwargs):
        return random.choice(list(board.legal_moves))

    def quit(self):
        pass

    def ucinewgame(self):
        pass

class RandomEngine(TourneyEngine):
    def __init__(self, engine = None, name = 'random', movetime = None, nodes = None, depth = None):
        super().__init__(_RandomEngineBackend(), name, movetime = movetime, nodes = nodes)

    def getMoveWithCP(self, board):
        return self.engine.play(board), 0

    def getMove(self, board):
        return self.engine.play(board)

    def getResults(self, board):
        return _Random_Results(self.engine.play(board))

class StockfishEngine(TourneyEngine):
    def __init__(self, movetime = _movetime, depth = 30, sfPath = _stockfishPath, engine = None, name = None):
        #self.skill = skill
        self.name = name
        self.stockfishPath = sfPath

        engine = chess.engine.SimpleEngine.popen_uci([self.stockfishPath], stderr = subprocess.PIPE)

        engine.configure({'UCI_AnalyseMode' : 'false'})

        super().__init__(engine, f'd{depth} {movetime}', movetime = movetime, depth = depth)

class LC0Engine(TourneyEngine):
    def __init__(self, weightsPath = None, nodes = None, movetime = _movetime, isHai = True, lc0Path = None, threads = 1, backend = 'blas', backend_opts = '', name = None, engine = None, noise = False, extra_flags = None, verbose = False, temperature = 0, temp_decay = 0):
        self.weightsPath = weightsPath
        self.lc0Path = lc0Path if lc0Path is not None else _lc0Path
        self.isHai = isHai
        self.threads = threads
        self.noise = noise
        self.verbose = verbose
        engine = chess.engine.SimpleEngine.popen_uci([self.lc0Path, f'--weights={weightsPath}', f'--threads={threads}', f'--backend={backend}', f'--backend-opts={backend_opts}', f'--temperature={temperature}', f'--tempdecay-moves={temp_decay}'] + (['--noise'] if self.noise else []) + ([f'--noise-epsilon={noise}'] if isinstance(self.noise, float) else [])+ (['--verbose-move-stats'] if self.verbose else []) + (extra_flags if extra_flags is not None else []), stderr=subprocess.DEVNULL)

        if name is None:
            name = f"{os.path.basename(self.weightsPath)[:-6]} {movetime}"
        super().__init__(engine, name, movetime = movetime, nodes = nodes)

class MaiaEngine(LC0Engine):
    def __init__(self, model_path, **kwargs):

        self.Model = Trained_Model(model_path)

        kwargs['weightsPath'] = self.Model.getMostTrained()

        super().__init__(**kwargs)

class OldMaiaEngine(LC0Engine):
    def __init__(self, model_path, **kwargs):
        kwargs['weightsPath'] = model_path
        super().__init__(lc0Path = 'lc0', **kwargs)

class NewMaiaEngine(LC0Engine):
    def __init__(self, model_path, **kwargs):
        kwargs['weightsPath'] = model_path
        super().__init__(lc0Path = 'lc0_23', **kwargs)

class HaibridEngine(LC0Engine):
    def __init__(self, **kwargs):
        super().__init__(**kwargs, isHai = True)

class LeelaEngine(LC0Engine):
    def __init__(self, **kwargs):
        super().__init__(**kwargs, isHai = False)

def playMixedGame(B1, G1, B2, G2, turns, round = None, startingFen = None, notes = None):
    timeStarted = datetime.datetime.now(tz)
    if startingFen is not None:
        board = chess.Board(fen=startingFen)
    else:
        board = chess.Board()

    players = [(B1, G1), (B2, G2)]
    i = 0
    while not board.is_game_over():
        B, G = players[i % 2]
        if i > 0 and i % turns == 0:
            E = G
        else:
            E = B
        board.push(E.getMove(board))
        i += 1
    pgnGame = chess.pgn.Game.from_board(board)

    pgnGame.headers['Event'] = f"{B1.name}*{G1.name} vs {B2.name}*{G2.name} at {turns} turns"
    pgnGame.headers['White'] = B1.name
    pgnGame.headers['White_helper'] = G1.name
    pgnGame.headers['Black'] = B2.name
    pgnGame.headers['Black_helper'] = G2.name
    pgnGame.headers['Date'] = timeStarted.strftime("%Y-%m-%d %H:%M:%S")
    pgnGame.headers['Turns'] = str(turns)
    pgnGame.headers['Mixed_game'] = "True"
    if round is not None:
        pgnGame.headers['Round'] = round
    if notes is not None:
        for k, v in notes.items():
            pgnGame.headers[k] = v
    return pgnGame

def playGame(E1, E2, round = None, maxMoves = None, startingFen = None, notes = None, startingMoves = None):

    timeStarted = datetime.datetime.now(tz)
    i = 0
    if startingFen is not None:
        board = chess.Board(fen=startingFen)
        if startingMoves is not None:
            for m in startingMoves:
                board.push_san(m)
                i += 1
    else:
        board = chess.Board()
    E1.newgame()
    E2.newgame()

    players = [E1, E2]

    while not board.is_game_over():
        E = players[i % 2]
        board.push(E.getMove(board))
        i += 1
        if maxMoves is not None and i > maxMoves * 2:
            break
    pgnGame = chess.pgn.Game.from_board(board)

    pgnGame.headers['Event'] = f"{E1.name} vs {E2.name}"
    pgnGame.headers['White'] = E1.name
    pgnGame.headers['Black'] = E2.name
    pgnGame.headers['Date'] = timeStarted.strftime("%Y-%m-%d %H:%M:%S")
    if round is not None:
        pgnGame.headers['Round'] = round
    if notes is not None:
        for k, v in notes.items():
            pgnGame.headers[k] = v
    return pgnGame

def playSequence(E1, E2, maxMoves, fen):
    movesDat = []
    board = chess.Board(fen=fen)
    players = [E1, E2]
    for i in range(maxMoves):
        for j in range(2):
            if board.is_game_over():
                break
            E = players[j]
            #import pdb; pdb.set_trace()
            results = E.getResults(board)
            movesDat.append([board.san(results.move), cpToInt(results.info.get('score'))])
            board.push(results.move)
    return movesDat

def getTrajectory(engine, game, return_boards = False, remove_history = False):
    b = game.board()

    emoves = []
    hmoves = []
    boards = []
    for i, hMov in enumerate(game.mainline_moves()):
        if i % 2 == 1:
            if remove_history:
                eMov = engine.getMove(chess.Board(b.fen()))
            else:
                eMov = engine.getMove(b)
            hmoves.append(hMov.uci())
            emoves.append(eMov.uci())
            boards.append(b.fen())
        b.push(hMov)
    if return_boards:
        return emoves, hmoves, boards
    else:
        return emoves, hmoves

def checkTrajectories(engineStr, gamesPath, resultsDir):
    E = stringToEngine(engineStr)

    games = []
    with open(gamesPath) as f:
        g = chess.pgn.read_game(f)
        while g is not None:
            games.append(g)
            g = chess.pgn.read_game(f)

    saveName = os.path.join(resultsDir, f"{json.loads(engineStr)['name']}-{os.path.basename(gamesPath)}")

    print(f"Starting: {saveName[-50:]}")

    with open(saveName, 'w') as f:
        for i, g in enumerate(games):
            engineT, humanT = getTrajectory(E, g)
            json.dump({
                'human' : humanT,
                'engine' : engineT,
                'site' : g.headers.get('Site', 'missing'),
            }, f)
            f.write('\n')
            f.flush()
            print(f"{saveName[-50:]} {i} games done")

def playSafeGame(E1, E2, E1str, E2str, round = None, maxMoves = None, startingFen = None, notes = None, swapPlayers = False, startingMoves = None):
    try:
        if swapPlayers:
            pgnGame = playGame(E2, E1, startingFen = startingFen, round = round, maxMoves = maxMoves, notes = notes, startingMoves = startingMoves)
        else:
            pgnGame = playGame(E1, E2, startingFen = startingFen, round = round, maxMoves = maxMoves, notes = notes, startingMoves = startingMoves)
    except BrokenPipeError:
        print("BrokenPipe: {E1.name} v {E2.name}")
        E1 = stringToEngine(E1str)
        E2 = stringToEngine(E2str)
        return playSafeGame(E1, E2, E1str, E2str, round = round, startingFen = startingFen, notes = notes, swapPlayers = swapPlayers, startingMoves = startingMoves)
    except chess.engine.EngineTerminatedError as e:
        print(f"engine.EngineTerminatedError, likely protobuf: {E1.name} v {E2.name}\n {e}")
        raise
    return pgnGame, E1, E2

def playSafeSequence(E1, E2, E1str, E2str, maxMoves, fen):
    try:
        movesdat = playSequence(E1, E2, maxMoves, fen)
    except BrokenPipeError:
        print("BrokenPipe: {E1.name} v {E2.name}")
        E1 = stringToEngine(E1str)
        E2 = stringToEngine(E2str)
        return playSafeSequence(E1, E2, E1str, E2str, maxMoves, fen)
    except chess.engine.EngineTerminatedError as e:
        print(f"engine.EngineTerminatedError, likely protobuf: {E1.name} v {E2.name}\n {e}")
        raise
    return movesdat, E1, E2

def playBoard(E, startingFen):
    board = chess.Board(fen=startingFen)
    move = E.getMove(board)
    return board.san(move)

def playSafeBoard(E, Estr, board):
    if isinstance(board, str):
        board = chess.Board(fen=board)
    try:
        move = E.getMove(board)
    except BrokenPipeError:
        print("BrokenPipe: {E.name}")
        E = stringToEngine(Estr)
        return playSafeBoard(E, Estr, startingFen)

def getBoardResults(E, board):
    if isinstance(board, str):
        board = chess.Board(fen=startingFen)

    results = E.getResults(board)
    return results, board.san(results.move)

def playSafeBoard(E, Estr, board):
    try:
        results, move = getBoardResults(E, board)
    except BrokenPipeError:
        print("BrokenPipe: {E.name}")
        E = stringToEngine(Estr)
        return playSafeBoard(E, Estr, board)
    except chess.engine.EngineTerminatedError as e:
        print(f"engine.EngineTerminatedError, likely protobuf: {E}\n {e}")
        raise
    return results, move, E

def playBoardStarts(Estr, boards, resultsDir, opponent = None, check_lines = True, suffix = None, opponentPlayDepth = 4):
    tstart = time.time()
    E = stringToEngine(Estr)
    eName = json.loads(Estr)['name']

    printWithDate(f"Starting {E} analysis", flush = True)

    if opponent is not None:
        Eop = stringToEngine(opponent)
        eOppName = json.loads(opponent)['name']
        printWithDate(f"Starting opponent {Eop}", flush = True)
        fvs = open(os.path.join(resultsDir, f"{eName}-v-{eOppName}{suffix if suffix is not None else ''}.json"), 'w', buffering = 1)

    with open(os.path.join(resultsDir, f"{eName}{suffix if suffix is not None else ''}.json"), 'w', buffering = 1) as f:
        for bDat in boards:
            dat = {'board' : bDat['board']}
            results, move, E = playSafeBoard(E, Estr, bDat['board'])
            dat['engine_move'] = move
            if move == bDat['safe'][0]:
                dat['engine_move_path'] = 'safe'
            elif move == bDat['tricky'][0]:
                dat['engine_move_path'] = 'tricky'
            else:
                dat['engine_move_path'] = 'neither'
            if 'score' in results.info:
                dat['engine_cp_start'] = cpToInt(results.info['score'])

            if check_lines:
                vsInfos = {}
                for s in ['safe', 'tricky']:
                    board = chess.Board(fen = bDat['board'])
                    board.push_san(bDat[s][0])
                    board.push_san(bDat[s][1])
                    results, move, E = playSafeBoard(E, Estr, board.fen())
                    dat[f'{s}_move'] = move
                    if bDat[s][2] == move:
                        dat[f'{s}_move_followed_path'] = True
                    else:
                        dat[f'{s}_move_followed_path'] = False
                    if 'score' in results.info:
                        dat[f'engine_cp_{s}'] = cpToInt(results.info['score'])
                    if opponent is not None:
                        board.push_san(move)
                        #swapping players because opponent is moving first
                        seqDat, Eop, E  = playSafeSequence(Eop, E , opponent, Estr, opponentPlayDepth, board.fen())
                        vsInfos[s] = [[move, cpToInt(results.info.get('score'))]] + seqDat
                if opponent is not None:
                    for s in ['safe', 'tricky']:
                        vsDat = {'board' : bDat['board'], 'line' : s}
                        for i, (m, cp) in enumerate(vsInfos[s]):
                            if i % 2:
                                vsDat[f'opponent_move_{i//2}'] = m
                                vsDat[f'opponent_cp_{i//2}'] = cp
                            else:
                                vsDat[f'engine_move_{i//2}'] = m
                                vsDat[f'engine_cp_{i//2}'] = cp
                        fvs.write(json.dumps(vsDat))
                        fvs.write('\n')
            f.write(json.dumps(dat))
            f.write('\n')
    if opponent is not None:
        fvs.close()
    printWithDate(f"Done {len(boards)} games in {time.time() - tstart : .2f}s of {eName}")

def playBoardSequence(E1str, E2str, boards, resultsDir, maxMoves = None, startSeq = None, notes = None, suffix = None):
    tstart = time.time()
    E1 = stringToEngine(E1str)

    E2 = stringToEngine(E2str)
    e1Name = json.loads(E1str)['name']
    e2Name = json.loads(E2str)['name']
    games = []
    print(f"Starting {E1.name} vs {E2.name}", flush = True)
    for bDat in boards:
        for i in range(2):
            if startSeq:
                pgnGame, E1, E2 = playSafeGame(E1, E2, E1str, E2str, startingFen = bDat['board'], round = i, maxMoves = maxMoves, startingMoves = bDat[startSeq][:2], swapPlayers = bool(i), notes = notes)
            else:
                pgnGame, E1, E2 = playSafeGame(E1, E2, E1str, E2str, startingFen = bDat['board'], round = i, maxMoves = maxMoves, swapPlayers = bool(i), notes = notes)

            with open(os.path.join(resultsDir, f"{e1Name}-{e2Name}{suffix if suffix is not None else ''}.pgn"), 'a') as f:
                pgnStr = str(pgnGame)
                f.write(pgnStr)
                f.write('\n\n')
                games.append(pgnStr)
    print(f"Done {len(boards)} games in {time.time() - tstart : .2f}s of {e1Name} vs {e2Name}")

    return games

def playMixedTourney(b1str, g1str, b2str, g2str, turns, num_rounds, resultsDir):
    tstart = time.time()

    B1 = stringToEngine(b1str)
    B2 = stringToEngine(b2str)
    G1 = stringToEngine(g1str)
    G2 = stringToEngine(g2str)

    b1Name = json.loads(b1str)['name']
    b2Name = json.loads(b2str)['name']
    g1Name = json.loads(g1str)['name']
    g2Name = json.loads(g2str)['name']

    games = []
    i = 0
    print(f"Starting {B1.name}*{G1.name} vs {B2.name}*{G2.name} at {turns} turns", flush = True)
    while i < num_rounds:
        try:
            if i % 2 == 0:
                players = [B1, G1, B2, G2]
            else:
                players = [B2, G2, B1, G1]
            pgnGame = playMixedGame(*players, turns, round = i + 1)
        except BrokenPipeError:
            print("BrokenPipe: {B1.name}*{G1.name} vs {B2.name}*{G2.name} at {turns} turns", flush = True)
            b1Name = json.loads(b1str)['name']
            b2Name = json.loads(b2str)['name']
            g1Name = json.loads(g1str)['name']
            g2Name = json.loads(g2str)['name']
            continue
        except chess.engine.EngineTerminatedError as e:
            print(f"engine.EngineTerminatedError, likely protobuf: {B1.name}*{G1.name} vs {B2.name}*{G2.name}\n {e}")
            raise
        else:
            pgnStr = str(pgnGame)
            with open(os.path.join(resultsDir, f"{b1Name}*{g1Name}-{b2Name}*{g2Name}.pgn"), 'a') as f:
                for game in games:
                    f.write(pgnStr)
                    f.write('\n\n')
            games.append(pgnStr)
            i += 1
    print(f"Done {num_rounds} games in {time.time() - tstart : .2f}s of {B1.name}*{G1.name} vs {B2.name}*{G2.name}  at {turns}")

    return games


def playTourney(E1str, E2str, num_rounds, resultsDir):
    tstart = time.time()
    E1 = stringToEngine(E1str)

    E2 = stringToEngine(E2str)
    e1Name = json.loads(E1str)['name']
    e2Name = json.loads(E2str)['name']
    games = []
    i = 0
    print(f"Starting {E1.name} vs {E2.name}", flush = True)
    while i < num_rounds:
        try:
            if i % 2 == 0:
                players = [E1, E2]
            else:
                players = [E2, E1]
            pgnGame = playGame(*players, round = i + 1)
        except BrokenPipeError:
            print("BrokenPipe: {E1.name} v {E2.name}")
            E1 = stringToEngine(E1str)
            E2 = stringToEngine(E2str)
            continue
        except chess.engine.EngineTerminatedError as e:
            print(f"engine.EngineTerminatedError, likely protobuf: {E1.name} v {E2.name}\n {e}")
            raise
        else:
            pgnStr = str(pgnGame)
            with open(os.path.join(resultsDir, f"{e1Name}-{e2Name}.pgn"), 'a') as f:
                f.write(pgnStr)
                f.write('\n\n')
            games.append(pgnStr)
            i += 1
    print(f"Done {num_rounds} games in {time.time() - tstart : .2f}s of {e1Name} vs {e2Name}")

    return games

def listRandoms():
    return [json.dumps({'engine' : 'random', 'config' : {}, 'name' : 'random'})]

def listLeelas(configs = None):
    if configs is None:
        configs = {}
    vals = []
    for e in os.scandir(os.path.join(networksDir, 'leela_weights')):
        if e.name.endswith('pb.gz'):
            v = {'weightsPath' : e.path}
            v.update(configs)
            vals.append(v)
    return [json.dumps({'engine' : 'leela', 'config' : v, 'name' : f"leela_{os.path.basename(v['weightsPath']).split('-')[1]}"}) for v in vals]

def listHaibrids(configs = None, netsDir = '', suffix = '-64x6-140000.pb.gz'):
    if configs is None:
        configs = {}
    vals = []
    for e in os.scandir(os.path.join(networksDir, netsDir)):
        if e.name.endswith(suffix):
            v = {'weightsPath' : e.path}
            v.update(configs)
            vals.append(v)
    return [json.dumps({'engine' : 'hiabrid', 'config' : v, 'name' : f"hiabrid_{os.path.basename(v['weightsPath']).split('-')[0]}"}) for v in vals]

def fileNameToEngineName(s):
    if 'stockfish' in s:
        n, s, m, d = s.split('_')
        return "StockfishEngine s{} d{} {}".format(s[:-1], d[:-1], m[:-1])
    elif 'leela' in s:
        n, e = s.split('_')
        return "LeelaEngine t3-{}".format(e)
    elif 'hiabrid' in s:
        n, e, *_ = s.split('_')
        return "HaibridEngine {}-64x6-140000".format(e)
    elif 'random' in s:
        return 'RandomEngine random'
    raise RuntimeError(f"{s} is not a valid engine file name")

def listStockfishs():
    vals = []
    for s, m, d in zip(stockfish_SKILL, stockfish_MOVETIMES, stockfish_DEPTHS):
        vals.append({
            'skill' : s,
            'movetime' : m,
            'depth' : d,
        })
    return [json.dumps({'engine' : 'stockfish', 'config' : v, 'name' : f"stockfish_{v['skill']}s_{v['movetime']}m_{v['depth']}d"}) for v in vals]

def stringToEngine(s):
    dat = json.loads(s)
    if dat['engine'] == 'lc0':
        return LC0Engine(**dat['config'])
    elif dat['engine'] == 'stockfish':
        return StockfishEngine(**dat['config'])
    elif dat['engine'] == 'hiabrid':
        return HaibridEngine(**dat['config'])
    elif dat['engine'] == 'leela':
        return LeelaEngine(**dat['config'])
    elif dat['engine'] == 'maia':
        return MaiaEngine(**dat['config'])
    elif dat['engine'] == 'maia_old':
        return OldMaiaEngine(**dat['config'])
    elif dat['engine'] == 'maia_new':
        return NewMaiaEngine(**dat['config'])
    elif dat['engine'] == 'random':
        return RandomEngine(**dat['config'])
    else:
        raise RuntimeError(f"Invalid config: {s}")

def playStockfishGauntlet(E, num_rounds):
    pgns = []
    for config in listStockfishs():
        SF = StockfishEngine(**config)
        p = playTourney(E, SF)
        pgns += p
    return pgns

def listAllEngines(hiabridConfig = None, leelaConfig = None):
    return listHaibrids(configs = hiabridConfig) + listLeelas(configs = leelaConfig) + listStockfishs() + listRandoms()
