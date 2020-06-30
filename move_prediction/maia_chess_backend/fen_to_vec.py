import re

import chess
import numpy as np

# Generate the regexs
boardRE = re.compile(r"(([^/]+)/([^/]+)/([^/]+)/([^/]+)/([^/]+)/([^/]+)/([^/]+)/([^/]+)) ((w)|(b)) ((-)|(K)?(Q)?(k)?(q)?)( ((-)|(\w+)))?( \d+)?( \d+)?")

replaceRE = re.compile(r'[1-8/]')

pieceMapWhite = {'E' : [False] * 12}
pieceMapBlack = {'E' : [False] * 12}

piece_reverse_lookup = {}

all_pieces = 'PNBRQK'

for i, p in enumerate(all_pieces):
    #White pieces first
    mP = [False] * 12
    mP[i] = True
    pieceMapBlack[p] = mP
    piece_reverse_lookup[i] = p

    #then black
    mP = [False] * 12
    mP[i + len(all_pieces)] = True
    pieceMapBlack[p.lower()] = mP
    piece_reverse_lookup[i  + len(all_pieces)] = p.lower()


    #Black pieces first
    mP = [False] * 12
    mP[i] = True
    pieceMapWhite[p.lower()] = mP

    #then white
    mP = [False] * 12
    mP[i + len(all_pieces)] = True
    pieceMapWhite[p] = mP

iSs = [str(i + 1) for i in range(8)]
eSubss = [('E' * i, str(i)) for i in range(8,0, -1)]
castling_vals = 'KQkq'

def toByteBuff(l):
    return b''.join([b'\1' if e else b'\0' for e in l])

pieceMapBin = {k : toByteBuff(v) for k,v in pieceMapBlack.items()}

def toBin(c):
    return pieceMapBin[c]

castlesMap = {True : b'\1'*64, False : b'\0'*64}

#Some previous lines are left in just in case

# using N,C,H,W format

move_letters = list('abcdefgh')

moves_lookup = {}
move_ind = 0
for r_1 in range(8):
    for c_1 in range(8):
        for r_2 in range(8):
            for c_2 in range(8):
                moves_lookup[f"{move_letters[r_1]}{c_1+1}{move_letters[r_2]}{c_2+1}"] = move_ind
                move_ind += 1

def move_to_index(move_str):
    return moves_lookup[move_str[:4]]

def array_to_preproc(a_target):
    if not isinstance(a_target, np.ndarray):
        #check if toch Tensor without importing torch
        a_target = a_target.cpu().numpy()
    if a_target.dtype != np.bool_:
        a_target = a_target.astype(np.bool_)
    piece_layers = a_target[:12]
    board_a = np.moveaxis(piece_layers, 2, 0).reshape(64, 12)
    board_str = ''
    is_white = bool(a_target[12, 0, 0])
    castling = [bool(l[0,0]) for l in a_target[13:]]
    board = [['E'] * 8 for i in range(8)]
    for i in range(12):
        for x in range(8):
            for y in range(8):
                if piece_layers[i,x,y]:
                    board[x][y] = piece_reverse_lookup[i]
    board = [''.join(r) for r in board]
    return ''.join(board), is_white, tuple(castling)

def preproc_to_fen(boardStr, is_white, castling):
    rows = [boardStr[(i*8):(i*8)+8] for i in range(8)]

    if not is_white:
        castling = castling[2:] + castling[:2]
        new_rows = []
        for b in rows:
            new_rows.append(b.swapcase()[::-1].replace('e', 'E'))

        rows = reversed(new_rows)
    row_strs = []
    for r in rows:
        for es, i in eSubss:
            if es in r:
                r = r.replace(es, i)
        row_strs.append(r)
    castle_str = ''
    for i, v in enumerate(castling):
        if v:
            castle_str += castling_vals[i]
    if len(castle_str) < 1:
        castle_str = '-'

    is_white_str = 'w' if is_white else 'b'
    board_str = '/'.join(row_strs)
    return f"{board_str} {is_white_str} {castle_str} - 0 1"

def array_to_fen(a_target):
    return preproc_to_fen(*array_to_preproc(a_target))

def array_to_board(a_target):
    return chess.Board(fen = array_to_fen(a_target))

def simple_fen_vec(boardStr, is_white, castling):
    castles = [np.frombuffer(castlesMap[c], dtype='bool').reshape(1, 8, 8) for c in castling]
    board_buff_map = map(toBin, boardStr)
    board_buff = b''.join(board_buff_map)
    a = np.frombuffer(board_buff, dtype='bool')
    a = a.reshape(8, 8, -1)
    a = np.moveaxis(a, 2, 0)
    if is_white:
        colour_plane = np.ones((1, 8, 8), dtype='bool')
    else:
        colour_plane = np.zeros((1, 8, 8), dtype='bool')

    return np.concatenate([a, colour_plane, *castles], axis = 0)

def preproc_fen(fenstr):
    r = boardRE.match(fenstr)
    if r.group(14):
        castling = (False, False, False, False)
    else:
        castling = (bool(r.group(15)), bool(r.group(16)), bool(r.group(17)), bool(r.group(18)))
    if r.group(11):
        is_white = True
        rows_lst = r.group(1).split('/')
    else:
        is_white = False
        castling = castling[2:] + castling[:2]
        rows_lst = r.group(1).swapcase().split('/')
        rows_lst = reversed([s[::-1] for s in rows_lst])

    rowsS = ''.join(rows_lst)
    for i, iS in enumerate(iSs):
        if iS in rowsS:
            rowsS = rowsS.replace(iS, 'E' * (i + 1))
    return rowsS, is_white, castling

def fenToVec(fenstr):
    return simple_fen_vec(*preproc_fen(fenstr))

def fenToVec_old(fenstr):
    r = boardRE.match(fenstr)
    if r.group(11):
        is_white = True
        pMap = pieceMapBlack
    else:
        is_white = False
        pMap = pieceMapWhite

    #rowsS = replaceRE.sub(replacer, r.group(1))
    rowsS = r.group(1).replace('/', '')
    for i, iS in enumerate(iSs):
        if iS in rowsS:
            rowsS = rowsS.replace(iS, 'E' * (i + 1))
    #rows = [v  for ch in rowsS for v in pieceMap[ch]]
    rows = []
    for c in rowsS:
        rows += pMap[c]

    a_b_img = np.swapaxes(np.swapaxes(np.array(rows, dtype='bool').reshape((8, 8, -1)), 0, 2), 2, 1)

    if not is_white:
        a_b_img = np.flip(a_b_img, axis = 1)
    #En passant, castling and 50 move counter need to be added
    extras = [is_white]
    #extras have 8 * 8 = 64 bits
    #import pdb; pdb.set_trace()
    if r.group(14):
        castling = [False, False, False, False]
    else:
        castling = [bool(r.group(15)), bool(r.group(16)), bool(r.group(17)), bool(r.group(18))]
    #print(castling)
    extras += castling

    extras += [False] * (64 - len(extras))
    extras_a = np.array(extras).reshape(-1, 8, 8)

    return np.append(a_b_img, extras_a, axis = 0)

def gameToVecs(game):
    boards = []
    board = game.board()
    for i, node in enumerate(game.mainline()):
        fen = str(board.fen())
        board.push(node.move)
        boards.append(fenToVec(fen))
    return np.stack(boards, axis = 0)
