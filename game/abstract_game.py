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

    start = 1
    ag = AbstractGame("tictactoe", False)
    ag.network.load("latest.pt")
    ag.mcts.mode = 'test'
    state = TicTacToe(is_render=True)
    state.reset(start)
    state.render(f"当前局面 {start}作为开始的玩家")
    v, p = ag.network.predict(state.get_torch_state())
    print(f"当前对于玩家 {state.get_current_player()}游戏,{v}, {p}，{np.argmax(p)}")

    print('=' * 120)
    state.move((0, 1))
    state.render(f"当前局面")
    v, p = ag.network.predict(state.get_torch_state())
    print(f"当前对于玩家 {state.get_current_player()}游戏,{v}, {p}，{np.argmax(p)}")

    print('=' * 120)
    state.move((0, 0))
    state.render(f"当前局面")
    v, p = ag.network.predict(state.get_torch_state())
    print(f"当前对于玩家 {state.get_current_player()}游戏,{v}, {p}，{np.argmax(p)}")

    print('=' * 120)
    state.move((1, 1))
    state.render(f"当前局面")
    v, p = ag.network.predict(state.get_torch_state())
    print(f"当前对于玩家 {state.get_current_player()}游戏,{v}, {p}，{np.argmax(p)}")

    print('=' * 120)
    state.move((1, 2))
    state.render(f"当前局面")
    v, p = ag.network.predict(state.get_torch_state())
    cs = state.get_torch_state()
    print(
        f"当前对于玩家 {state.get_current_player()}游戏,{v}, {p}，{np.argmax(p)} "
        f"\n\n当前状态为\n{cs[:, :, 0]}\n {cs[:, :, 1]}\n\n {cs[:, :, 2]}")

    visit_list = ag.mcts.get_action_probability(state=state, is_greedy=False)
    print(f"当前对于玩家 {state.get_current_player()},mcts的概率为 {visit_list}, {np.argmax(visit_list)}")
