import chess.uci

import collections
import concurrent.futures
import threading

import re
import os.path

probRe = re.compile(r"\(P: +([^)]+)\)")


class ProbInfoHandler(chess.uci.InfoHandler):
    def __init__(self):
        super().__init__()
        self.info["probs"] = []

    def on_go(self):
        """
        Notified when a *go* command is beeing sent.

        Since information about the previous search is invalidated, the
        dictionary with the current information will be cleared.
        """
        with self.lock:
            self.info.clear()
            self.info["refutation"] = {}
            self.info["currline"] = {}
            self.info["pv"] = {}
            self.info["score"] = {}
            self.info["probs"] = []

    def string(self, string):
        """Receives a string the engine wants to display."""
        prob = re.search(probRe, string).group(1)
        self.info["probs"].append(string)

class EngineHandler(object):
    def __init__(self, engine, weights, threads = 2):
        self.enginePath = os.path.normpath(engine)
        self.weightsPath = os.path.normpath(weights)

        self.engine = chess.uci.popen_engine([self.enginePath, "--verbose-move-stats", f"--threads={threads}", f"--weights={self.weightsPath}"])

        self.info_handler = ProbInfoHandler()
        self.engine.info_handlers.append(self.info_handler)

        self.engine.uci()
        self.engine.isready()

    def __repr__(self):
        return f"<EngineHandler {self.engine.name} {self.weightsPath}>"

    def getBoardProbs(self, board, movetime = 1000, nodes = 1000):
        self.engine.ucinewgame()
        self.engine.position(board)
        moves = self.engine.go(movetime = movetime, nodes = nodes)
        probs = self.info_handler.info['probs']
        return moves, probs
