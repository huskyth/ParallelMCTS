from game.chess.chess import Chess
from game.tictactoe.tictactoe import TicTacToe
from mcts.pure_mcts import MCTS
from models.tictactoe.network_wrapper import TictactoeNetWrapper
from models.wm_model.network_wrapper import ChessNetWrapper


class AbstractGame:
    def __init__(self, name):
        if name == 'WMChess':
            self._network = ChessNetWrapper()
            self._start_epoch = self._network.try_load()
            self._mcts = MCTS(self._network.predict, mode='train', name="AI")
            self._random_network = ChessNetWrapper()
            self._random_mcts = MCTS(self._random_network.predict, mode='test', name="随机玩家")
            self._state = Chess()
        elif name == "tictactoe":
            self._network = TictactoeNetWrapper()
            self._start_epoch = self._network.try_load()
            self._mcts = MCTS(self._network.predict, mode='train', name="AI")
            self._random_network = TictactoeNetWrapper()
            self._random_mcts = MCTS(self._random_network.predict, mode='test', name="随机玩家")
            self._state = TicTacToe()
        else:
            raise ValueError("Invalid game name")

    @property
    def mcts(self):
        return self._mcts

    @property
    def random_mcts(self):
        return self._random_mcts

    @property
    def start_epoch(self):
        return self._start_epoch

    @property
    def network(self):
        return self._network

    @property
    def state(self):
        return self._state
