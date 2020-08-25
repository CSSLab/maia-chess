#!/usr/bin/env python3
import sys
import numpy as np
from .policy_index import policy_index

columns = 'abcdefgh'
rows = '12345678'
promotions = 'rbq' # N is encoded as normal move

col_index = {columns[i] : i for i in range(len(columns))}
row_index = {rows[i] : i for i in range(len(rows))}

def index_to_position(x):
    return columns[x[0]] + rows[x[1]]

def position_to_index(p):
    return col_index[p[0]], row_index[p[1]]

def valid_index(i):
    if i[0] > 7 or i[0] < 0:
        return False
    if i[1] > 7 or i[1] < 0:
        return False
    return True

def queen_move(start, direction, steps):
    i = position_to_index(start)
    dir_vectors = {'N': (0, 1), 'NE': (1, 1), 'E': (1, 0), 'SE': (1, -1),
            'S':(0, -1), 'SW':(-1, -1), 'W': (-1, 0), 'NW': (-1, 1)}
    v = dir_vectors[direction]
    i = i[0] + v[0] * steps, i[1] + v[1] * steps
    if not valid_index(i):
        return None
    return index_to_position(i)

def knight_move(start, direction, steps):
    i = position_to_index(start)
    dir_vectors = {'N': (1, 2), 'NE': (2, 1), 'E': (2, -1), 'SE': (1, -2),
            'S':(-1, -2), 'SW':(-2, -1), 'W': (-2, 1), 'NW': (-1, 2)}
    v = dir_vectors[direction]
    i = i[0] + v[0] * steps, i[1] + v[1] * steps
    if not valid_index(i):
        return None
    return index_to_position(i)

def make_map(kind='matrix'):
    # 56 planes of queen moves
    moves = []
    for direction in ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']:
        for steps in range(1, 8):
            for r0 in rows:
                for c0 in columns:
                    start = c0 + r0
                    end = queen_move(start, direction, steps)
                    if end == None:
                        moves.append('illegal')
                    else:
                        moves.append(start+end)

    # 8 planes of knight moves
    for direction in ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']:
        for r0 in rows:
            for c0 in columns:
                start = c0 + r0
                end = knight_move(start, direction, 1)
                if end == None:
                    moves.append('illegal')
                else:
                    moves.append(start+end)

    # 9 promotions
    for direction in ['NW', 'N', 'NE']:
        for promotion in promotions:
            for r0 in rows:
                for c0 in columns:
                    # Promotion only in the second last rank
                    if r0 != '7':
                        moves.append('illegal')
                        continue
                    start = c0 + r0
                    end = queen_move(start, direction, 1)
                    if end == None:
                        moves.append('illegal')
                    else:
                        moves.append(start+end+promotion)

    for m in policy_index:
        if m not in moves:
            raise ValueError('Missing move: {}'.format(m))

    az_to_lc0 = np.zeros((80*8*8, len(policy_index)), dtype=np.float32)
    indices = []
    legal_moves = 0
    for e, m in enumerate(moves):
        if m == 'illegal':
            indices.append(-1)
            continue
        legal_moves += 1
        # Check for missing moves
        if m not in policy_index:
            raise ValueError('Missing move: {}'.format(m))
        i = policy_index.index(m)
        indices.append(i)
        az_to_lc0[e][i] = 1

    assert legal_moves == len(policy_index)
    assert np.sum(az_to_lc0) == legal_moves
    for e in range(80*8*8):
        for i in range(len(policy_index)):
            pass
    if kind == 'matrix':
        return az_to_lc0
    elif kind == 'index':
        return indices

if __name__ == "__main__":
    # Generate policy map include file for lc0
    if len(sys.argv) != 2:
        raise ValueError("Output filename is needed as a command line argument")

    az_to_lc0 = np.ravel(make_map('index'))
    header = \
"""/*
 This file is part of Leela Chess Zero.
 Copyright (C) 2019 The LCZero Authors

 Leela Chess is free software: you can redistribute it and/or modify
 it under the terms of the GNU General Public License as published by
 the Free Software Foundation, either version 3 of the License, or
 (at your option) any later version.

 Leela Chess is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 GNU General Public License for more details.

 You should have received a copy of the GNU General Public License
 along with Leela Chess.  If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

namespace lczero {
"""
    line_length = 12
    with open(sys.argv[1], 'w') as f:
        f.write(header+'\n')
        f.write('const short kConvPolicyMap[] = {\\\n')
        for e, i in enumerate(az_to_lc0):
            if e % line_length == 0 and e > 0:
                f.write('\n')
            f.write(str(i).rjust(5))
            if e != len(az_to_lc0)-1:
                f.write(',')
        f.write('};\n\n')
        f.write('}  // namespace lczero')
