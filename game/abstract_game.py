from mcts.pure_mcts import MCTS
from models.wm_model.network_wrapper import ChessNetWrapper


class AbstractGame:
    def __init__(self, name):
        if name == 'WMChess':
            self._network = ChessNetWrapper()
            self._start_epoch = self._network.try_load()
            self._mcts = MCTS(self._network.predict, mode='test')
            self._random_network = ChessNetWrapper()
            self._random_mcts = MCTS(self._random_network.predict, mode='test')

        elif name == "tictactoe":
            pass
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
