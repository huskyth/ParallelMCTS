import copy

import numpy as np
import torch

from chess.common import MOVE_LIST, MOVE_TO_INDEX_DICT, INDEX_TO_MOVE_DICT, from_array_to_input_tensor

LEFT_RIGHT_POINT_MAP = {
    0: 0, 1: 3, 3: 1, 2: 2, 4: 8, 8: 4, 6: 10, 10: 6, 5: 9, 9: 5, 16: 16, 17: 18, 18: 17,
    20: 20, 19: 19, 12: 12, 13: 14, 14: 13, 15: 15, 7: 11, 11: 7
}
LEFT_POINT = [1, 4, 6, 7, 13, 5, 17]
RIGHT_POINT = [LEFT_RIGHT_POINT_MAP[l] for l in LEFT_POINT]
VERTICAL_MIDDLE_POINT = [0, 2, 16, 20, 19, 12, 15]
LEFT_RIGHT_ACTION_MAP = {}
LEFT_ACTION_INDEX = []
RIGHT_ACTION_INDEX = []
for index, action in enumerate(MOVE_LIST):
    from_idx, to_idx = action
    f, t = LEFT_RIGHT_POINT_MAP[from_idx], LEFT_RIGHT_POINT_MAP[to_idx]
    idx = MOVE_TO_INDEX_DICT[(f, t)]
    if index == idx:
        LEFT_RIGHT_ACTION_MAP[index] = idx
    else:
        LEFT_RIGHT_ACTION_MAP[index] = idx
        LEFT_RIGHT_ACTION_MAP[idx] = index
for index, action in enumerate(MOVE_LIST):
    from_idx, to_idx = action
    if from_idx in LEFT_POINT + VERTICAL_MIDDLE_POINT and to_idx in LEFT_POINT + VERTICAL_MIDDLE_POINT:
        if from_idx in VERTICAL_MIDDLE_POINT and to_idx in VERTICAL_MIDDLE_POINT:
            continue
        LEFT_ACTION_INDEX.append(index)
        RIGHT_ACTION_INDEX.append(LEFT_RIGHT_ACTION_MAP[index])

print()
TOP_BOTTOM_POINT_MAP = {0: 15, 15: 0, 1: 13, 13: 1, 2: 12, 12: 2, 3: 14, 14: 3,
                        4: 7, 7: 4, 8: 11, 11: 8, 6: 6, 5: 5, 17: 17, 20: 20, 18: 18, 9: 9, 10: 10,
                        16: 19, 19: 16
                        }
TOP_POINT = [0, 1, 2, 3, 4, 8, 16]
BOTTOM_POINT = [TOP_BOTTOM_POINT_MAP[t] for t in TOP_POINT]
HORIZONTAL_MIDDLE_POINT = [6, 5, 17, 20, 18, 9, 10]
TOP_BOTTOM_ACTION_MAP = {}
TOP_ACTION_INDEX = []
BOTTOM_ACTION_INDEX = []
for index, action in enumerate(MOVE_LIST):
    from_idx, to_idx = action
    f, t = TOP_BOTTOM_POINT_MAP[from_idx], TOP_BOTTOM_POINT_MAP[to_idx]
    idx = MOVE_TO_INDEX_DICT[(f, t)]
    if index == idx:
        TOP_BOTTOM_ACTION_MAP[index] = idx
    else:
        TOP_BOTTOM_ACTION_MAP[index] = idx
        TOP_BOTTOM_ACTION_MAP[idx] = index
for index, action in enumerate(MOVE_LIST):
    from_idx, to_idx = action
    if from_idx in TOP_POINT + HORIZONTAL_MIDDLE_POINT and to_idx in TOP_POINT + HORIZONTAL_MIDDLE_POINT:
        if from_idx in HORIZONTAL_MIDDLE_POINT and to_idx in HORIZONTAL_MIDDLE_POINT:
            continue
        TOP_ACTION_INDEX.append(index)
        BOTTOM_ACTION_INDEX.append(TOP_BOTTOM_ACTION_MAP[index])
print()


def lr(board, pi, current_player):
    new_board = np.ascontiguousarray(np.fliplr(board))
    assert id(board) != id(new_board)

    l, r = np.array(LEFT_ACTION_INDEX), np.array(RIGHT_ACTION_INDEX)
    new_pi = copy.deepcopy(pi)
    new_pi[l], new_pi[r] = new_pi[r], new_pi[l]
    new_current_player = current_player

    return new_board, new_pi, new_current_player


def tb_(board, pi, current_player):
    new_board = np.ascontiguousarray(np.flipud(board))
    assert id(board) != id(new_board)
    t, b = np.array(TOP_ACTION_INDEX), np.array(BOTTOM_ACTION_INDEX)
    new_pi = copy.deepcopy(pi)
    new_pi[t], new_pi[b] = new_pi[b], new_pi[t]
    new_current_player = current_player

    return new_board, new_pi, new_current_player


def board_to_torch_state(board, player):
    temp = torch.from_numpy(board).unsqueeze(0).unsqueeze(0)
    state0 = (temp > 0).float()
    state1 = (temp < 0).float()
    if player == -1:
        state0, state1 = state1, state0
    return torch.cat([state0, state1], dim=1)
