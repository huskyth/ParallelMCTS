import os

import cv2

from game.chess.chess_board import ChessBoard
from game.chess.common import from_array_to_input_tensor, GAME_MAP, MOVE_TO_INDEX_DICT, INDEX_TO_MOVE_DICT

from constants import ROOT_PATH

debug_path = ROOT_PATH / "debug"
if not debug_path.exists():
    debug_path.mkdir()
SCREEN_WIDTH = 580
SCREEN_HEIGHT = 580
CHESSMAN_WIDTH = 20
CHESSMAN_HEIGHT = 20
BLACK = 1
WHITE = -1


class Chess(ChessBoard):
    def __init__(self, start_player=1):
        super().__init__()
        self.current_player = start_player
        self.move_to_index = MOVE_TO_INDEX_DICT
        self.index_to_move = INDEX_TO_MOVE_DICT

    def is_end(self):
        winner = self.check_winner()
        is_end = winner is not None
        return is_end, winner

    @staticmethod
    def _fix_xy(target):
        x = GAME_MAP[target][0] * \
            SCREEN_WIDTH - CHESSMAN_WIDTH * 0.5
        y = GAME_MAP[target][1] * \
            SCREEN_HEIGHT - CHESSMAN_HEIGHT * 1
        return x, y

    def dump(self, key):
        image = cv2.imread(str(ROOT_PATH / "chess/assets/watermelon.png"))
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
        cv2.imwrite(str(debug_path) + os.sep + key + '.png', image)

    def get_torch_state(self):
        """
            得到棋盘的张量
            :return:
        """
        return from_array_to_input_tensor(self.pointStatus, self.current_player)

    def do_action(self, action):
        self.execute_move(action, self.current_player)
        self.current_player *= -1

    def get_current_player(self):
        return self.current_player

    def reset(self, start_player=1):
        self.init_point_status()
        self.current_player = start_player

    def render(self, key):
        pass