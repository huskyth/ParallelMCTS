import copy

import cv2
import numpy as np
import torch

from game.chess.chess_board import ChessBoard
from game.chess.common import from_array_to_input_tensor, GAME_MAP, MOVE_TO_INDEX_DICT, INDEX_TO_MOVE_DICT

from constants import ROOT_PATH
from game.chess.symmetry_creator import lr, tb_, LEFT_ACTION_INDEX, RIGHT_ACTION_INDEX, TOP_ACTION_INDEX, \
    BOTTOM_ACTION_INDEX

debug_path = ROOT_PATH / "debug"
if not debug_path.exists():
    debug_path.mkdir()
SCREEN_WIDTH = 580
SCREEN_HEIGHT = 580
CHESSMAN_WIDTH = 20
CHESSMAN_HEIGHT = 20
BLACK = 1
WHITE = -1
MAX_DRAW_TIME = 5


class Chess(ChessBoard):
    def __init__(self, start_player=1, is_render=False):
        self.current_player = start_player
        super().__init__()
        self.move_to_index = MOVE_TO_INDEX_DICT
        self.index_to_move = INDEX_TO_MOVE_DICT
        self.is_render = is_render
        self.last_action = (-1, -1)

    def is_end(self, mock=False):
        winner = self.check_winner(mock)
        is_end = winner is not None
        return is_end, winner

    @staticmethod
    def _fix_xy(target):
        x = GAME_MAP[target][0] * \
            SCREEN_WIDTH - CHESSMAN_WIDTH * 0.5
        y = GAME_MAP[target][1] * \
            SCREEN_HEIGHT - CHESSMAN_HEIGHT * 1
        return x, y

    def _write_point(self):
        image = cv2.imread(str(ROOT_PATH / "game/chess/assets/watermelon.png"))
        for index, point in enumerate(self.pointStatus):
            if point == 0:
                continue
            (x, y) = Chess._fix_xy(index)
            if point == BLACK:
                cv2.circle(img=image, color=(0.0, 0.0, 0.0),
                           center=(int(x + CHESSMAN_WIDTH / 2), int(y + CHESSMAN_HEIGHT / 2)),
                           radius=int(CHESSMAN_HEIGHT // 2 * 1.5), thickness=-1)
            elif point == WHITE:
                cv2.circle(img=image, color=(255.0, 0.0, 0.0),
                           center=(int(x + CHESSMAN_WIDTH / 2), int(y + CHESSMAN_HEIGHT / 2)),
                           radius=int(CHESSMAN_HEIGHT // 2 * 1.5), thickness=-1)
        return image

    def render(self, key):
        if not self.is_render:
            return
        print(f"当前局面{self.pointStatus}的日志如下\n{key}\n")

    def center_probability(self, pi):
        l, r = np.array(LEFT_ACTION_INDEX), np.array(RIGHT_ACTION_INDEX)
        new_pi = copy.deepcopy(pi)
        new_pi[l], new_pi[r] = new_pi[r], new_pi[l]
        t, b = np.array(TOP_ACTION_INDEX), np.array(BOTTOM_ACTION_INDEX)
        new_pi = copy.deepcopy(new_pi)
        new_pi[t], new_pi[b] = new_pi[b], new_pi[t]
        return new_pi

    def get_torch_state(self):
        """
            得到棋盘的张量
            :return:
        """
        state = from_array_to_input_tensor(self.pointStatus, self.current_player, self.last_action)
        return state

    def do_action(self, action):
        self.execute_move(action, self.current_player)
        self.current_player *= -1

        # str_point = [str(t) for t in self.pointStatus] + [str(self.get_current_player())]
        # str_point = "".join(str_point)
        #
        # if str_point not in self.draw_checker:
        #     self.draw_checker[str_point] = 1
        # else:
        #     self.draw_checker[str_point] += 1
        #     if self.draw_checker[str_point] == MAX_DRAW_TIME:
        #         self.draw_checker['has'] = True

    def get_current_player(self):
        return self.current_player

    def reset(self, start_player=1):
        self.init_point_status()
        self.current_player = start_player
        self.last_action = (-1, -1)
        # self.reset_draw_checker()

    def move_random(self):
        import random
        l_move = self.get_legal_moves(self.get_current_player())
        l_move = random.choice(l_move)
        max_act = self.move_to_index[l_move]
        return max_act

    def top_buttom(self, s, p):
        board = s
        pi = p
        new_board, new_pi = tb_(board, pi)
        if isinstance(new_board, np.ndarray):
            new_board = torch.from_numpy(new_board).float()
        if isinstance(new_pi, np.ndarray):
            new_pi = torch.from_numpy(new_pi).float()
        return new_board, new_pi

    def image_show(self, key, is_image_show, wait_key=1):
        if not is_image_show:
            return
        img = self._write_point()
        cv2.imshow(key, img)
        return cv2.waitKey(wait_key)

    def left_right(self, s, p):
        board = s
        pi = p
        new_board, new_pi = lr(board, pi)
        if isinstance(new_board, np.ndarray):
            new_board = torch.from_numpy(new_board).float()
        if isinstance(new_pi, np.ndarray):
            new_pi = torch.from_numpy(new_pi).float()
        return new_board, new_pi

    def center(self, s, p):
        board = s
        pi = p
        new_board, new_pi = lr(board, pi)
        new_board, new_pi = tb_(new_board, new_pi)
        if isinstance(new_board, np.ndarray):
            new_board = torch.from_numpy(new_board).float()
        if isinstance(new_pi, np.ndarray):
            new_pi = torch.from_numpy(new_pi).float()

        return new_board, new_pi


if __name__ == '__main__':
    import os
    print(os.name)
    s = Chess().get_torch_state()
    print(s[:, :, 0])
    print(s[:, :, 1])
    print(s[:, :, 2])
    print(s[:, :, 3])
