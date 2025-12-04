import threading

import pygame
import os

import torch

os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'
import numpy as np

from .common import shiftOutChessman, DISTANCE, GAME_MAP
import copy

from .common import MOVE_TO_INDEX_DICT

BLACK = 1
WHITE = -1
SCREEN_WIDTH = 580
SCREEN_HEIGHT = 580
CHESSMAN_WIDTH = 20
CHESSMAN_HEIGHT = 20


class WMChessGUI:
    def __init__(self, mcts_player, play_state, human_color=-1, fps=6):

        # screen
        self.board = None
        self.width = 580
        self.height = 580

        self.fps = fps

        # human color
        self.human_color = human_color

        # reset items
        self.board = None
        self.number = None
        self.is_human = None
        self.human_move = None
        self.chessman_in_hand = None
        self.chosen_chessman_color = None
        self.chosen_chessman = None

        # reset status
        self.reset_status()

        self.mcts_player = mcts_player
        self.play_state = play_state

    def __del__(self):
        # close window
        self.is_running = False

    def init_point_status(self):
        self.board = []
        black = [0, 1, 2, 3, 4, 8]
        white = [7, 11, 12, 13, 14, 15]
        for x in range(21):
            self.board.append(0)
        for x in black:
            self.board[x] = BLACK
        for x in white:
            self.board[x] = WHITE
        self.board = np.array(self.board)

    # reset status
    def reset_status(self):
        self.init_point_status()

        self.is_human = False
        self.human_move = -1

        self.chessman_in_hand = False
        self.chosen_chessman_color = None
        self.chosen_chessman = None

    # human play
    def set_is_human(self, value=True):
        self.is_human = value

    def get_is_human(self):
        return self.is_human

    def get_human_move(self):
        return self.human_move

    def get_human_color(self):
        return self.human_color

    # execute move
    def execute_move(self, color, move, info=None):
        from_int, to_int = move
        print(f"🌿 exec {from_int} to {to_int}")
        assert color == WHITE or color == BLACK
        assert self.board[from_int] == color
        assert self.board[to_int] == 0
        assert DISTANCE[from_int][to_int] == 1
        self.board[from_int] = 0
        self.board[to_int] = color
        bake_point_status = copy.deepcopy(self.board)
        self.board = shiftOutChessman(
            bake_point_status, DISTANCE)

    def start(self):
        t = threading.Thread(target=self.loop)
        t.start()

    # main loop
    def loop(self):
        # set running
        self.is_running = True

        # init
        pygame.init()
        self.screen = pygame.display.set_mode((self.width, self.height))

        pygame.display.set_caption("WmChess")

        # timer
        self.clock = pygame.time.Clock()

        # background image
        base_folder = os.path.dirname(__file__)
        self.background_img = pygame.image.load(
            os.path.join(base_folder, 'assets/watermelon.png')).convert()

        # font
        self.font = pygame.font.SysFont('Arial', 16)

        while self.is_running:
            # timer
            self.clock.tick(self.fps)

            # handle event
            for event in pygame.event.get():
                # close window
                if event.type == pygame.QUIT:
                    self.is_running = False

                # human play
                if self.is_human:
                    if event.type == pygame.MOUSEBUTTONDOWN:
                        print("🌿 Mouse button down")
                        mouse_x, mouse_y = pygame.mouse.get_pos()
                        chessman = self._chosen_chessman(mouse_x, mouse_y)
                        if chessman is None:
                            continue
                        if not self.chessman_in_hand:
                            if self.board[chessman] == self.human_color:
                                self.chosen_chessman_color = self.board[chessman]
                                self.chessman_in_hand = True
                                self.chosen_chessman = chessman
                                print(f"🌿 chessman in hand human_color is {self.human_color}")

                        else:
                            if self.board[chessman] == 0 and \
                                    DISTANCE[self.chosen_chessman][chessman] == 1:

                                self.human_move = (self.chosen_chessman, chessman)
                                self.execute_move(self.human_color, self.human_move, "人类玩家")
                                self.human_move = MOVE_TO_INDEX_DICT[self.human_move]
                                self.set_is_human(False)
                                self.play_state.do_action(self.human_move)
                            else:
                                self.board[
                                    self.chosen_chessman] = self.chosen_chessman_color
                            self.chessman_in_hand = False
                            # draw

                else:
                    self.play_state.is_render = False
                    self.mcts_player.update_tree(-1)
                    pi = self.mcts_player.get_action_probability(self.play_state, False)
                    move_idx = np.argmax(pi)
                    state = self.play_state.get_torch_state()
                    v, p = self.mcts_player.predict(state)
                    print(f"当前玩家 {self.play_state.get_current_player()} 的 MCTS 模拟概率为:\n\n {pi} \n\n "
                          f"直接预测的价值为 {v} \n\n"
                          f"直接预测的概率为 \n\n {p} \n\n"
                          f"直接预测会选择的行为 {np.argmax(p)} ，蒙特卡洛预测行为 {move_idx}\n\n"
                          f"当前第一维度：\n\n {state[:, :, 0]}\n\n\n"
                          f"当前第二维度：\n\n {state[:, :, 1]}\n\n\n"
                          f"当前第三维度：\n\n {state[:, :, 2]}\n\n\n")
                    move = self.play_state.index_to_move[move_idx]
                    self.execute_move(-self.human_color, move, info="AI")
                    self.play_state.do_action(move)
                    self.is_human = True

                # draw
                self._draw_background()
                self._draw_chessman()

            # refresh
            pygame.display.flip()

    def _chosen_chessman(self, x, y):
        x, y = x / (SCREEN_WIDTH + 0.0), y / (SCREEN_HEIGHT + 0.0)
        for point in range(21):
            if abs(x - GAME_MAP[point][0]) < 0.05 and abs(y - GAME_MAP[point][1]) < 0.05:
                print(f"🌿 choose {point}")
                return point
        return None

    def _draw_background(self):
        # load background
        self.screen.blit(self.background_img, (0, 0))

    @staticmethod
    def fix_xy(target):
        x = GAME_MAP[target][0] * \
            SCREEN_WIDTH - CHESSMAN_WIDTH * 0.5
        y = GAME_MAP[target][1] * \
            SCREEN_HEIGHT - CHESSMAN_HEIGHT * 1
        return x, y

    def draw_end_string(self, string):
        text_surface = self.font.render(string, True, (255, 255, 0))
        self.screen.blit(text_surface, (500, 560))
        pygame.display.update()

    def _draw_chessman(self):
        for index, point in enumerate(self.board):
            if point == 0:
                continue
            (x, y) = WMChessGUI.fix_xy(index)
            if point == BLACK:
                pygame.draw.circle(self.screen, (0, 0, 0), (int(x + CHESSMAN_WIDTH / 2), int(y + CHESSMAN_HEIGHT / 2)),
                                   int(CHESSMAN_HEIGHT // 2 * 1.5))
            elif point == WHITE:
                pygame.draw.circle(self.screen, (255, 0, 0),
                                   (int(x + CHESSMAN_WIDTH / 2), int(y + CHESSMAN_HEIGHT / 2)),
                                   int(CHESSMAN_HEIGHT // 2 * 1.5))
