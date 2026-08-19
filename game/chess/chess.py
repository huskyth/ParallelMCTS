import copy
from collections import deque

import cv2
import numpy as np
import torch

from game.chess.chess_board import ChessBoard
from game.chess.common import from_array_to_input_tensor, GAME_MAP, MOVE_TO_INDEX_DICT, INDEX_TO_MOVE_DICT, \
    MAX_HISTORY_STEPS

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
MAX_DRAW_TIME = 4


class Chess(ChessBoard):
    def __init__(self, start_player=1, is_render=False):
        self.current_player = start_player
        super().__init__()
        self.move_to_index = MOVE_TO_INDEX_DICT
        self.index_to_move = INDEX_TO_MOVE_DICT
        self.is_render = is_render
        self.last_action = deque(maxlen=MAX_HISTORY_STEPS)

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

    # def _write_point(self):
    #     image = cv2.imread(str(ROOT_PATH / "game/chess/assets/watermelon.png"))
    #     for index, point in enumerate(self.pointStatus):
    #         if point == 0:
    #             continue
    #         (x, y) = Chess._fix_xy(index)
    #         if point == BLACK:
    #             cv2.circle(img=image, color=(0.0, 0.0, 0.0),
    #                        center=(int(x + CHESSMAN_WIDTH / 2), int(y + CHESSMAN_HEIGHT / 2)),
    #                        radius=int(CHESSMAN_HEIGHT // 2 * 1.5), thickness=-1)
    #         elif point == WHITE:
    #             cv2.circle(img=image, color=(0.0, 0.0, 255.0),
    #                        center=(int(x + CHESSMAN_WIDTH / 2), int(y + CHESSMAN_HEIGHT / 2)),
    #                        radius=int(CHESSMAN_HEIGHT // 2 * 1.5), thickness=-1)
    #     return image

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
        one_time = np.sum(np.array(self.pointStatus) == 1)
        neg_one_time = np.sum(np.array(self.pointStatus) == -1)

        self.execute_move(action, self.current_player)
        self.current_player *= -1

        str_point = [str(t) for t in self.pointStatus] + [str(self.get_current_player())]
        str_point = "".join(str_point)

        if str_point not in self.draw_checker:
            self.draw_checker[str_point] = 1
        else:
            self.draw_checker[str_point] += 1
            if self.draw_checker[str_point] == MAX_DRAW_TIME:
                self.draw_checker['has'] = True

        one_time_after = np.sum(np.array(self.pointStatus) == 1)
        neg_one_time_after = np.sum(np.array(self.pointStatus) == -1)
        assert abs(one_time_after - one_time) == 0 or abs(neg_one_time - neg_one_time_after) == 0
        return abs(one_time_after - one_time) + abs(neg_one_time - neg_one_time_after)

    def get_current_player(self):
        return self.current_player

    def reset(self, start_player=1):
        self.init_point_status()
        self.current_player = start_player
        self.last_action = deque(maxlen=MAX_HISTORY_STEPS)
        self.reset_draw_checker()
        self.turn = 0

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

    def draw_policy_info(self, image, pi, player=None, top_k=15, step_reward=None, return_val=None):
        """
        在图像的右下角绘制策略信息。
        pi: 长度为 num_actions 的概率数组 (numpy array)
        player: 当前玩家（1 或 -1），用于显示在左上角
        top_k: 显示概率最大的前 k 个动作
        step_reward: 当前步的即时奖励（吃子奖励）
        return_val: 当前步的累积回报（z）
        """
        if pi is None:
            return image
        pi = np.array(pi)
        h, w = image.shape[:2]

        # ----- 在左上角显示当前玩家 -----
        if player is not None:
            player_text = "Black" if player == 1 else "White"
            cv2.putText(image, f"Player: {player_text}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

        # ----- 右下角策略信息框 -----
        overlay = image.copy()
        box_width = 250
        # 高度增加两行用于显示 step_reward 和 return
        box_height = 180 + 20 * (top_k + 1)
        x0 = w - box_width - 10
        y0 = h - box_height - 10
        cv2.rectangle(overlay, (x0, y0), (x0 + box_width, y0 + box_height), (255, 255, 255), -1)
        cv2.addWeighted(overlay, 0.8, image, 0.2, 0, image)
        cv2.rectangle(image, (x0, y0), (x0 + box_width, y0 + box_height), (0, 0, 0), 1)

        # 计算熵
        eps = 1e-8
        entropy = -np.sum(pi * np.log(pi + eps))
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        color_text = (0, 0, 0)
        thickness = 1

        y = y0 + 20
        cv2.putText(image, f"Entropy: {entropy:.3f}", (x0 + 5, y), font, font_scale, color_text, thickness)

        # 显示 step_reward 和 return（如果提供）
        if step_reward is not None:
            y += 20
            cv2.putText(image, f"Step Reward: {step_reward:.3f}", (x0 + 5, y), font, font_scale, color_text, thickness)
        if return_val is not None:
            y += 20
            cv2.putText(image, f"Return: {return_val:.3f}", (x0 + 5, y), font, font_scale, color_text, thickness)

        # Top-k 动作
        y += 20
        cv2.putText(image, "Top actions:", (x0 + 5, y), font, font_scale, color_text, thickness)

        indices = np.argsort(pi)[::-1][:top_k]
        for i, idx in enumerate(indices):
            y += 20
            prob = pi[idx]
            cv2.putText(image, f"  {idx}: {prob:.3f}", (x0 + 5, y), font, font_scale, color_text, thickness)

        return image

    def _write_point(self, pi=None, player=None, step_reward=None, return_val=None):
        """
        绘制棋盘，可选传入策略 pi、当前玩家、即时奖励和累积回报。
        """
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
                cv2.circle(img=image, color=(0.0, 0.0, 255.0),
                           center=(int(x + CHESSMAN_WIDTH / 2), int(y + CHESSMAN_HEIGHT / 2)),
                           radius=int(CHESSMAN_HEIGHT // 2 * 1.5), thickness=-1)
        # 如果传入了策略等信息，绘制信息框
        if pi is not None or player is not None:
            image = self.draw_policy_info(image, pi, player=player,
                                          step_reward=step_reward, return_val=return_val)
        return image

    def image_show(self, key, is_image_show, wait_key=5, pi=None, player=None,
                   step_reward=None, return_val=None):
        """
        显示图像，可传入策略 pi、当前玩家、即时奖励和累积回报。
        """
        if not is_image_show:
            return
        img = self._write_point(pi=pi, player=player,
                                step_reward=step_reward, return_val=return_val)
        cv2.imshow(key, img)
        return cv2.waitKey(wait_key)


if __name__ == '__main__':
    s = [0, -1,  0,  0,  1,  1,  1,  1,  0,  0,  1, -1,  1, -1,  0, -1,  0,
  0,  0,  0,  0.]
    c = Chess()
    c.pointStatus = s
    print(len(c.pointStatus))
    pi = [0.004635749850422144, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.004514929838478565, 0.0, 0.0046821278519928455, 0.0,
          0.0, 0.9525789618492126, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.004794583655893803, 0.004272001795470715,
          0.005039438139647245, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
          0.005404447205364704, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
          0.0, 0.0, 0.0, 0.0, 0.0, 0.004634215496480465, 0.004692332819104195, 0.004751227796077728, 0.0, 0.0, 0.0, 0.0,
          0.0, 0.0, 0.0, 0.0]
    c.image_show("a", True, 0, pi=pi, player=1)

    abv = [1, 2, 3, 4]
    print(np.random.shuffle(abv))
    print(abv)
    import os

    op = np.array([0.1, 0.14, 0.4, 0, 0, 0.34, 0.07, 0.05, 0, 0.05, 0, 0, 0, 0.03])
    print(np.sum(op))
    sta = Chess()

    legal_moves = list(sta.get_legal_moves(sta.get_current_player()))
    noise = 0.1 * np.random.dirichlet(0.03 * np.ones(np.count_nonzero(legal_moves)))

    prob = 0.9 * op
    j = 0
    for i in range(len(prob)):
        if legal_moves[i] == 1:
            prob[i] += noise[j]
            j += 1
    prob /= np.sum(prob)

    # sta.do_action((15, 12))
    # sta.do_action((0, 3))
    # sta.do_action((13, 15))
    # sta.do_action((1, 0))
    # sta.do_action((11, 10))

    # print(os.name)
    # s = sta.get_torch_state()
    # print(s[:, :, 0])
    # print(s[:, :, 1])
    # print(s[:, :, 2])
    # print(s[:, :, 3])
    # print(s[:, :, 4])
    # print(s[:, :, 5])
    # print(s[:, :, 6])
    # print(s[:, :, 7])
    # print(s[:, :, 8])
    # print(s[:, :, 9])
    # print(s[:, :, 10])
