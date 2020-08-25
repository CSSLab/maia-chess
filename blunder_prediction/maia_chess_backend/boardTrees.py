import json
import chess
import bz2
import multiprocessing
import functools
import networkx as nx
import matplotlib.pyplot as plt

import numpy as np



class BoardTree(object):
    def __init__(self, treeFile, lookupFile = None):

        if isinstance(treeFile, dict):
            self.root = treeFile
        else:
            self.root = loadbzJson(treeFile)
        if lookupFile is None:
            self.manualLookup = True
            self._lookup = {}
        elif isinstance(lookupFile, dict):
            self.manualLookup = False
            self._lookup = lookupFile
        else:
            self.manualLookup = False
            self._lookup = loadbzJson(lookupFile)

    def lookup(self, seq):
        try:
            return self._lookup[','.join(seq)]
        except KeyError:
            if self.manualLookup:
                k, f, c = makeFEN(seq)
                self._lookup[k] = (f,c)
                return (f,c)
            else:
                raise

    def getSeqs(self, depth):
        return genSeqs(self.root, depth)

    def getSeqSpread(self, seq, withMoves = False):
        moves = []
        values = []
        node = self.seqToNode(seq)
        for k, v in node.items():
            if k != 'hits' and k != 'sfeval':
                try:
                    values.append(float(v['sfeval']))
                except ValueError:
                    if '-' in v['sfeval']:
                        values.append(-10.0)
                    else:
                        values.append(10.0)
                moves.append(k)
        if len(seq) % 2:
            values = [v * -1 for v in values]
        if withMoves:
            return sorted(list(zip(values, moves)))
        else:
            return sorted(values)

    def seqToNode(self, seq):
        return nodeFromSeq(self.root, seq)

    def __getitem__(self, key):
        #Lists as keys, much advanced
        return self.seqToNode(key)

    def isTricky(self, seq, diff):
        spread = self.getSeqSpread(seq, withMoves=True)
        topVal = spread[-1][0]
        secVal = spread[-2][0]
        if topVal - secVal < diff or secVal > 10:
            return False, 0
        else:
            return spread[-1][1], topVal - secVal

    def isSafe(self, seq, diff):
        spread = self.getSeqSpread(seq, withMoves=True)
        topVal = spread[-1][0]
        secVal = spread[-2][0]
        if topVal - secVal < diff:
            return spread[-1][1], topVal - secVal
        else:
            return False, 0

    def checkLine(self, seq, checkFunc, depth, diff):
        depth = depth - 1
        try:
            move, delta =  checkFunc(seq, diff)
        except IndexError:
            return False, 0
        if not move:
            return False, 0
        trickSeq = seq + [move]
        if depth < 1:
            return trickSeq, delta
        try:
            spreadOP = self.getSeqSpread(trickSeq, withMoves=True)
            trickSeq.append(spreadOP[-1][1])
        except IndexError:
            return False, 0

        return self.checkLine(trickSeq, checkFunc, depth - 1, diff)

    def isTrickyLine(self, seq, depth, diff):
        return self.checkLine(seq, self.isTricky, depth, diff)

    def isSafeLine(self, seq, depth, diff):
        return self.checkLine(seq, self.isSafe, depth, diff)

    def getSeqInfos(self, seq):
        ret = {
            'moves' : seq,
            'hits' : [],
            'evals' : [],
            'num_children' : [],
            'possible_moves' : [],
        }
        for i in range(len(seq)):
            node = self.seqToNode(seq[:i + 1])
            info = self.lookup(seq[:i + 1])
            ret['hits'].append(node['hits'])
            ret['evals'].append(node['sfeval'])
            ret['num_children'].append(len(node.keys()) - 2)
            ret['possible_moves'].append(info[1])
        return ret

    def isStart(self, seq, depth, diff):
        spread = self.getSeqSpread(seq, withMoves=True)
        trickies = []
        safes = []
        safeExtras = {
            'deltas' : [],
            'hits' : [],
            'evals' : [],
            'num_children' : [],
            'possible_moves' : [],
        }
        trickyExtras = {
            'deltas' : [],
            'hits' : [],
            'evals' : [],
            'num_children' : [],
            'possible_moves' : [],
        }
        for v, m in spread[-5:]:
            opSpread = self.getSeqSpread(seq + [m], withMoves=True)
            for ov, om in opSpread[-3:]:
                trickVal, trickDelta = self.isTrickyLine(seq + [m] + [om], depth, diff)
                if trickVal:
                    trickies.append(trickVal)
                    trickyExtras['deltas'].append(trickDelta)
                    info = self.getSeqInfos(trickVal)
                    for k in ['hits', 'evals', 'num_children', 'possible_moves']:
                        trickyExtras[k].append(info[k])
                safeVal, safeDelta = self.isSafeLine(seq + [m] + [om], depth, diff)
                if safeVal:
                    safes.append(safeVal)
                    safeExtras['deltas'].append(safeDelta)
                    info = self.getSeqInfos(safeVal)
                    for k in ['hits', 'evals', 'num_children', 'possible_moves']:
                        safeExtras[k].append(info[k])
        d = {
            'tricky' : [t[len(seq):] for t in trickies],
            'tricky_deltas' : trickyExtras['deltas'],
            'safe': [t[len(seq):] for t in safes],
            'safe_deltas' : safeExtras['deltas'],
            'sequence' : seq,
            'fen' : chess.Board(self.lookup(seq)[0]).fen(),
            'lichess' : self.seqToLichess(seq),
            }
        for k in ['hits', 'evals', 'num_children', 'possible_moves']:
            d[f"tricky_{k}"] = [t[len(seq):] for t in trickyExtras[k]]
            d[f"safe_{k}"] = [t[len(seq):] for t in safeExtras[k]]
        return d

    def seqToLichess(self, seq):
        board = chess.Board(self.lookup(seq)[0])
        return "https://lichess.org/analysis/standard/" + board.fen().replace(' ', '_')

    def addNodes(self, g, rootName, seq, depth = None, evalFactor = 1):
        if depth < 1:
            return g
        node = self[seq]
        for k, v in list(node.items()):
            if k == 'hits' or k == 'sfeval':
                continue
            g.add_node(f"{rootName},{k}", hits = v['hits'], sfeval = v['sfeval'] * evalFactor, label = k)
            g.add_edge(rootName, f"{rootName},{k}", weight = v['hits'])
            if len(v) > 2:
                g = self.addNodes(g, f"{rootName},{k}", seq + [k], depth = depth - 1 if depth else None, evalFactor = evalFactor)
        return g

    def drawSeqTree(self, seq, ax = None, depth = 4, rootName = 'root', minHits = None, showLabels = True, labelDepth = 2, scalingFactor = .01):
        node = self[seq]
        evalFactor = -1 if len(seq) % 2 else 1
        G = nx.DiGraph()
        G.add_node(rootName, hits = node['hits'], sfeval = node['sfeval'] * evalFactor, label = rootName)
        G = self.addNodes(G, rootName, seq, depth=depth)

        if minHits is not None:
            for n, dat in list(G.nodes(data = True)):
                if dat['hits'] < minHits:
                    G.remove_node(n)

        shells = [[rootName]]
        done = set([rootName])
        new_neighbours = list(G.neighbors(rootName))
        done |= set(new_neighbours)


        while len(new_neighbours):
            shells.append(list(new_neighbours))
            new_new_neighbours = set()
            for n in new_neighbours:
                new_new_neighbours |= (set(G.neighbors(n)) - done)
            new_neighbours = new_new_neighbours

        topn = []
        for i in range(labelDepth):
            topn += shells[i]
        try:
            pos = nx.drawing.nx_agraph.graphviz_layout(G, prog='dot')

            meanX = np.mean([p1 for p1,p2 in pos.values()])
            pos = {k : ((p1 - meanX) * scalingFactor * (1 if k not in topn else 1.5), p2 * scalingFactor) for k, (p1, p2) in pos.items()}
        except ImportError:
            pos = nx.shell_layout(G, nlist = shells)
            pos = nx.spring_layout(G, pos=pos, fixed = shells[0] + shells[1], iterations = 100, k = .005)
        #pos = nx.spring_layout(G, iterations = 10, k = .005)

        nx.draw_networkx(G,
                        pos,
                        ax = ax,
                        node_size =  [d['hits'] + 200 for n,d in G.nodes(data = True)],
                        node_color =  [int(d['sfeval'].replace('#', '')) * 10 if isinstance(d['sfeval'], str) else d['sfeval'] for n,d in G.nodes(data = True)],
                        #font_color = 'xkcd:purple',
                        font_size = 16,
                        cmap = plt.get_cmap('bwr'),
                        vmin = -10,
                        vmax = 10,
                        labels = {n: d['label'] if n in topn else '' for n, d in G.nodes(data = True)},
                        with_labels = showLabels,
                        )
        return pos

def nodeFromSeq(root, seq):
    node = root
    for s in seq:
        node = node[s]
    return node

def genSeqs(startNode, depth):
    retSeqs = [[]]
    if depth < 1:
        return retSeqs
    for k, v in list(startNode.items()):
        if k == 'hits' or k == 'sfeval':
            pass
        elif len(v) <= 2:
            retSeqs.append([k])
        else:
            retSeqs +=  [[k] + s for s in genSeqs(v, depth - 1)]
    return retSeqs

def loadbzJson(filename):
    with bz2.open(filename, 'rt') as f:
        d = json.load(f)
    return d

def moveSeqToBoard(seq):
    board = chess.Board()
    for m in seq:
        board.push_san(m.replace('?', '').replace('!', ''))
    return board

def makeFEN(seq):
    board = moveSeqToBoard(seq)
    return ','.join(seq), board.fen(), len(list(board.legal_moves))
