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
        self.game = name
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
    import numpy as np

    start = 1
    ag = AbstractGame("WMChess", False)
    ag.network.load("latest.pt")
    ag.mcts.mode = 'test'

    state = Chess(is_render=True)

    # latest.pt
    # 模型已经加载
    # best.pt
    # 模型已经加载
    # 当前对于玩家
    # 1
    # 游戏, [[-0.07566565]], [8.98275321e-05 1.61475356e-04 7.78718459e-05 2.45539792e-04
    #                         7.82548013e-05 7.73889988e-05 7.83692958e-05 7.56562149e-05
    #                         8.44648675e-05 2.24890842e-04 8.06955359e-05 7.83648866e-05
    #                         7.91743732e-05 7.87651370e-05 6.97311480e-05 8.01141505e-05
    #                         1.95852335e-04 8.31803016e-04 8.45344912e-05 8.15639432e-05
    #                         7.34223577e-05 7.47420709e-05 1.09073961e-04 1.28362760e-01
    #                         8.57805135e-05 8.35234241e-05 7.70996339e-05 1.84925782e-04
    #                         7.96371387e-05 8.31016077e-05 2.69051321e-04 2.31401835e-04
    #                         7.84710792e-05 1.20029887e-04 8.31318903e-05 1.01768033e-04
    #                         8.34135950e-01 7.97047614e-05 8.18190892e-05 7.88905236e-05
    #                         9.27389119e-05 7.91004859e-05 2.85324715e-02 9.09001756e-05
    #                         7.86293676e-05 1.27911905e-03 7.76927045e-05 7.82809293e-05
    #                         1.01432299e-04 4.50562598e-04 1.14338625e-04 1.06827472e-04
    #                         7.87561949e-05 7.92008796e-05 7.76461893e-05 7.78533577e-05
    #                         8.97496939e-05 7.83354408e-05 7.77310925e-05 7.84041258e-05
    #                         7.91962739e-05 7.84966032e-05 8.20949208e-05 7.85515658e-05
    #                         7.84254371e-05 7.85422817e-05 8.55718317e-05 8.33203158e-05
    #                         7.85375596e-05 7.79895709e-05 7.50900799e-05 7.59370450e-05]，36
    state.reset(start)
    state.render(f"当前局面 {start}作为开始的玩家")
    v, p = ag.network.predict(state.get_torch_state())
    if start == 1:
        p = state.center_probability(p)
    print(f"当前对于玩家 {state.get_current_player()}游戏,{v}, {p}，{np.argmax(p)}")

    assert False

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
