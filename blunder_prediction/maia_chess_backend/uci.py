import chess.engine

import re
import os.path

probRe = re.compile(r"\(P: +([^)]+)\)")


class EngineHandler(object):
    def __init__(self, engine, weights, threads = 2):
        self.enginePath = os.path.normpath(engine)
        self.weightsPath = os.path.normpath(weights)

        self.engine = chess.engine.SimpleEngine.popen_uci([self.enginePath, "--verbose-move-stats", f"--threads={threads}", f"--weights={self.weightsPath}"])

    def __repr__(self):
        return f"<EngineHandler {self.engine.id['name']} {self.weightsPath}>"

    def getBoardProbs(self, board, movetime = 1000, nodes = 1000):
        probs = []
        with self.engine.analysis(board, chess.engine.Limit(time = movetime / 1000, nodes = nodes), info = chess.engine.INFO_ALL) as analysis:
            for info in analysis:
                if 'string' in info:
                    probs.append(info['string'])
        move = analysis.multipv[0].get('pv', [None])[0]
        return move, probs
