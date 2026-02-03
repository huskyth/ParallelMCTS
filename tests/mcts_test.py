import numpy as np

from game.chess.chess import Chess
from game.chess.common import INDEX_TO_MOVE_DICT
from mcts.pure_mcts import MCTS
from models.wm_model.network_wrapper import ChessNetWrapper
from utils.debug_tools import visualize_mcts

net = ChessNetWrapper()
mcts = MCTS(net.predict, 'test')

state = Chess(-1)

state.pointStatus = [1, 1, -1, 1, 1, 0, 0, 0, 0, -1, 0, 0, -1, 0, -1, 0, 0, -1, 1, 1, -1]

state.image_show("test", True, wait_key=0)

pi = mcts.get_action_probability(state, True)

move_idx = np.argmax(pi)
print(INDEX_TO_MOVE_DICT[move_idx])

# 最终局面
state.execute_move(INDEX_TO_MOVE_DICT[move_idx], state.get_current_player())
state.image_show("test", True, wait_key=0)

print(state.is_end())
