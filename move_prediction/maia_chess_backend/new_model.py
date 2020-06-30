import chess
import chess.engine

class ChessEngine(object):
    def __init__(self, engine, limits):
        self.limits = chess.engine.Limit(**limits)
        self.engine = engine

    def getMove(self, board):
        try:
            results = self.engine.play(
                    board,
                    limit=self.limits,
                    info = chess.engine.INFO_ALL
            )

        if isinstance(board, str):
            board
