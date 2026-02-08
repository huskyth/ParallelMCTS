# import numpy as np
#
# from game.chess.chess import Chess
# from game.chess.common import INDEX_TO_MOVE_DICT
# from mcts.pure_mcts import MCTS
# from models.wm_model.network_wrapper import ChessNetWrapper
#
# net = ChessNetWrapper()
# net.load("best.pt")
# mcts = MCTS(net.predict, 'test', simulate_times=80)
#
# state = Chess(1)
#
# state.pointStatus = [1, 1, -1, 1, 1, 0, 0, 0, 0, -1, 0, 0, -1, 0, -1, 0, 0, -1, 1, 1, -1]
#
# state.image_show("test", True, wait_key=0)
#
# pi = mcts.get_action_probability(state, True)
#
# move_idx = np.argmax(pi)
# print(INDEX_TO_MOVE_DICT[move_idx])
#
# # 最终局面
# state.execute_move(INDEX_TO_MOVE_DICT[move_idx], state.get_current_player())
# state.image_show("test", True, wait_key=0)
#
# print(state.is_end())

from models.tictactoe.network_wrapper import TictactoeNetWrapper
from pickle import Pickler, Unpickler
import sys
from utils.logger import Logger

sys.stdout = Logger()
with open(TictactoeNetWrapper.MODEL_SAVE_PATH / "train_history.examples", "rb") as f:
    temp = Unpickler(f).load()
import numpy as np

n = 0
for t in temp:
    n += len(t)
    for i, ti in enumerate(t):
        state, pi, cp, r = ti
        if True:
            print(f"第{i // 4}步骤")
            st = (ti[0][:, :, 0] * cp + ti[0][:, :, 1] * -cp).data.numpy()
            if abs(st[1, 1]) == 1 and len(np.argwhere(st != 0)) == 2 and (abs(st[0, 0]) == 1 or
                                                         abs(st[0, 2]) == 1 or
                                                         abs(st[2, 0]) == 1 or
                                                         abs(st[2, 2]) == 1):
                state_b, pi_b, cp_b, r_b = t[i - 4]
                befo = (state_b[:, :, 0] * cp_b + state_b[:, :, 1] * -cp_b).data.numpy()
                if (abs(befo[0, 0]) == 1 or
                                                         abs(befo[0, 2]) == 1 or
                                                         abs(befo[2, 0]) == 1 or
                                                         abs(befo[2, 2]) == 1) and len(np.argwhere(befo != 0)) == 1:
                    pass
            print(st)
            print(f"策略: {pi}, 奖励: {r}, 当前玩家: {cp}， max a {np.argmax(pi)}")
            print("=" * 120)

print(f"一共{n}步")
