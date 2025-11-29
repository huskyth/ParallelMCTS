from game.chess.chess import Chess
from game.tictactoe.tictactoe import TicTacToe
from mcts.pure_mcts import MCTS
from models.tictactoe.network_wrapper import TictactoeNetWrapper
from models.wm_model.network_wrapper import ChessNetWrapper


class AbstractGame:
    def __init__(self, name, is_render):
        if name == 'WMChess':
            self._network = ChessNetWrapper()
            self._random_network = ChessNetWrapper()
            self._state = Chess(is_render=is_render)
        elif name == "tictactoe":
            self._network = TictactoeNetWrapper()
            self._random_network = TictactoeNetWrapper()
            self._state = TicTacToe(is_render=is_render)
        else:
            raise ValueError("Invalid game name")
        self._start_epoch = self._network.try_load()
        self._random_mcts = MCTS(self._random_network.predict, mode='test', name="随机玩家")
        self._mcts = MCTS(self._network.predict, mode='train', name="AI")

    @property
    def mcts(self):
        return self._mcts

    @property
    def random_mcts(self):
        return self._random_mcts

    @property
    def random_network(self):
        return self._random_network

    @property
    def start_epoch(self):
        return self._start_epoch

    @property
    def network(self):
        return self._network

    @property
    def state(self):
        return self._state


if __name__ == '__main__':
    import torch
    import numpy as np
    ag = AbstractGame("tictactoe", False)
    ag.network.load("best.pt")
    state = TicTacToe(is_render=True)
    state.reset(-1)
    state.render(f"当前局面 {-1}作为开始的玩家")
    v, p = ag.network.predict(state.get_torch_state())
    print(f"当前游戏,{v}, {p}，{np.argmax(p)}")

    print('=' * 120)
    ag = AbstractGame("tictactoe", False)
    ag.network.load("best.pt")
    state = TicTacToe(is_render=True)
    state.reset(1)
    state.render(f"当前局面 {1}作为开始的玩家")
    v, p = ag.network.predict(state.get_torch_state())
    print(f"当前游戏,{v}, {p}，{np.argmax(p)}")
