import re

import chess
import numpy as np

# Generate the regexs
boardRE = re.compile(r"(([^/]+)/([^/]+)/([^/]+)/([^/]+)/([^/]+)/([^/]+)/([^/]+)/([^/]+)) ((w)|(b)) ((-)|(K)?(Q)?(k)?(q)?) (((-)|(\w+))) (\d+) (\d+)")


replaceRE = re.compile(r'[1-8/]')

all_pieces = 'pPnNbBrRqQkK'

"""
def replacer(matchObj):
    try:
        return 'E' * int(matchObj.group(0))
    except ValueError:
        return ''
"""

#Map pieces to lists

pieceMapWhite = {'E' : [False] * 12}
pieceMapBlack = {'E' : [False] * 12}

for i, p in enumerate(all_pieces):
    mP = [False] * 12
    mP[i] = True
    pieceMapWhite[p] = mP
    mP = [False] * 12
    mP[i + -1 if i % 2 else 1] = True
    pieceMapBlack[p] = mP

iSs = [str(i + 1) for i in range(i)]

#Some previous lines are left in just in case

def fenToVec(fenstr):
    r = boardRE.match(fenstr)
    if r.group(11):
        is_white = [True]
    else:
        is_white = [False]
    if r.group(14):
        castling = [False, False, False, False]
    else:
        castling = [bool(r.group(15)), bool(r.group(16)), bool(r.group(17)), bool(r.group(18))]

    #En passant and 50 move counter need to be added
    #rowsS = replaceRE.sub(replacer, r.group(1))
    rowsS = r.group(1).replace('/', '')
    for i, iS in enumerate(iSs):
        if iS in rowsS:
            rowsS = rowsS.replace(iS, 'E' * (i + 1))
    #rows = [v  for ch in rowsS for v in pieceMap[ch]]
    rows = []
    for c in rowsS:
        rows += pieceMap[c]
    return np.array(rows + castling + is_white, dtype='bool')

def fenToVec(fenstr):
    r = boardRE.match(fenstr)
    if r.group(11):
        pMap = pieceMapBlack
    else:
        pMap = pieceMapWhite
    """
    if r.group(14):
        castling = [False, False, False, False]
    else:
        castling = [bool(r.group(15)), bool(r.group(16)), bool(r.group(17)), bool(r.group(18))]
    """
    #rowsS = replaceRE.sub(replacer, r.group(1))
    rowsS = r.group(1).replace('/', '')
    for i, iS in enumerate(iSs):
        if iS in rowsS:
            rowsS = rowsS.replace(iS, 'E' * (i + 1))
    #rows = [v  for ch in rowsS for v in pieceMap[ch]]
    rows = []
    for c in rowsS:
        rows += pMap[c]
    #En passant, castling and 50 move counter need to be added
    return np.array(rows, dtype='bool').reshape((8,8,12))
