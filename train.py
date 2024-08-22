import os
import sys
import threading
import time
from collections import deque

import numpy as np

from chess.common import ROOT_PATH, INDEX_TO_MOVE_DICT, MODEL_SAVE_PATH, MOVE_TO_INDEX_DICT, draw_chessmen, \
    draw_chessman_from_image, ANALYSIS_PATH, create_directory
from symmetry_creator import lr, tb_, board_to_torch_state
from tensor_board_tool import MySummary

path = str(ROOT_PATH / "chess")
if path not in sys.path:
    sys.path.append(path)

from chess.chess import Chess
from chess.wm_chess_gui import WMChessGUI
from mcts.parallel_mcts import MCTS
from network_wrapper import ChessNetWrapper
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import pickle


class Trainer:
    def __init__(self):
        self.epoch = 100
        self.test_rate = 5
        self.greedy_times = 5
        self.dirichlet_rate = 0.1
        self.dirichlet_probability = 0.3
        self.contest_number = 8
        self.use_gui = True
        self.network = ChessNetWrapper()
        self.old_network = ChessNetWrapper()
        self.mcts = MCTS(self.network.predict)
        self.state = Chess()
        self.train_sample = deque(maxlen=10240)
        self.wm_chess_gui = WMChessGUI(7, -1)
        self.writer = MySummary(use_wandb=True)

    def _load(self):
        multiprocessing.set_start_method("spawn")
        print(f"process start method {multiprocessing.get_start_method()}")
        if os.path.exists(str(MODEL_SAVE_PATH / "old_version.pt")) and os.path.exists(
                str(MODEL_SAVE_PATH / "checkpoint.example")):
            print("load from old pth...")
            self.network.load("old_version.pt")
            self.load_samples()

    def _collect(self):
        return self._play()

    def get_symmetries(self, board, pi, current_player):
        ret = [(board, pi, current_player, "origin")]
        new_board, new_pi, new_current_player = \
            lr(board, pi, current_player)
        ret.append((new_board, new_pi, new_current_player, "lr"))

        new_board, new_pi, new_current_player = \
            tb_(board, pi, current_player)
        ret.append((new_board, new_pi, new_current_player, "tb"))

        new_board_1, new_pi_1, new_current_player_1 = \
            lr(new_board, new_pi, new_current_player)
        ret.append((new_board_1, new_pi_1, new_current_player_1, "center"))
        return ret

    def _play(self):
        step = 0
        train_sample = []
        self.mcts.update_tree(-1)
        self.state.reset()
        if self.use_gui:
            self.wm_chess_gui.reset_status()
        while not self.state.is_end()[0]:
            step += 1
            is_greedy = step > self.greedy_times
            probability = self.mcts.get_action_probability(state=self.state, is_greedy=is_greedy)

            temp = self.get_symmetries(self.state.get_board(), probability, self.state.get_current_player())
            # TODO://如有问题，一起测试这里
            for board, pi, current_player, _ in temp:
                board = board_to_torch_state(board, current_player)
                train_sample.append([board, pi, current_player])

            legal_action = self.state.get_legal_moves(self.state.get_current_player())
            legal_action = [MOVE_TO_INDEX_DICT[x] for x in legal_action]
            dirichlet_noise = self.dirichlet_rate * np.random.dirichlet(
                self.dirichlet_probability * np.ones(len(legal_action)))
            probability = (1 - self.dirichlet_rate) * probability
            for i in range(len(legal_action)):
                probability[legal_action[i]] += dirichlet_noise[i]
            probability = probability / probability.sum()
            action = np.random.choice(len(probability), p=probability)
            if self.use_gui:
                self.wm_chess_gui.execute_move(self.state.get_current_player(), INDEX_TO_MOVE_DICT[action])
            self.state.do_action(action)
            self.mcts.update_tree(action)

        self.writer.add_float(y=step, title="Training episode length")
        _, winner = self.state.is_end()
        assert winner is not None
        for item in train_sample:
            if item[-1] == winner:
                item.append(1)
            else:
                item.append(-1)

        return train_sample

    def contest(self, n):
        self.old_network.load("old_version.pt")
        new_win, old_win, draws = 0, 0, 0
        with ProcessPoolExecutor(max_workers=os.cpu_count()) as ppe:
            future_list = [
                ppe.submit(Trainer._contest, 1 if i % 2 == 0 else -1, self.network, self.old_network, self.state,
                           i) for i in range(n)]
            for item in as_completed(future_list):
                n, o, d = item.result()
                new_win += n
                old_win += o
                draws += d

        return new_win, old_win, draws

    @staticmethod
    def _contest(current_player, network, old_network, state, i):
        random_state = np.random.RandomState(i)
        start_player = current_player
        new_player = MCTS(network.predict)
        old_mcts = MCTS(old_network.predict)
        new_player.update_tree(-1)
        old_mcts.update_tree(-1)
        state.reset()
        player_list = [old_mcts, None, new_player]
        step = 0
        while not state.is_end()[0]:
            step += 1
            player = player_list[current_player + 1]
            probability_new = player.get_action_probability(state, True)
            max_act = random_state.choice(len(probability_new), p=probability_new)
            state.do_action(max_act)
            new_player.update_tree(max_act)
            old_mcts.update_tree(max_act)
            current_player *= -1
            if step >= 450:
                return 0, 0, 1

        del network
        del old_network

        _, winner = state.is_end()
        assert winner is not None
        if winner == 1:
            new_win = 1 if start_player == 1 else 0
        else:
            new_win = 1 if start_player == -1 else 0

        return new_win, 1 - new_win, 0

    def learn(self):
        self._load()
        if self.use_gui:
            t = threading.Thread(target=self.wm_chess_gui.loop)
            t.start()
        for epoch in range(self.epoch):
            self.writer.add_float(epoch, "Epoch")
            train_sample = self._collect()
            self.train_sample += train_sample
            self.network.save("old_version.pt")
            self.save_samples()
            if len(self.train_sample) >= 512:
                np.random.shuffle(self.train_sample)
                self.network.train(self.train_sample, self.writer)

            if (epoch + 1) % self.test_rate == 0:
                new_win, old_win, draws = self.contest(self.contest_number)
                all_ = new_win + old_win + draws
                self.writer.add_float(y=new_win, title="New player winning number")
                self.writer.add_float(y=old_win, title="Old player winning number")
                self.writer.add_float(y=draws, title="Draws number")
                self.writer.add_float(y=new_win / self.contest_number, title="Winning rate")

                if new_win / all_ > 0.6:
                    print("ACCEPT")
                    self.network.save("best.pt")
                else:
                    print("REJECT")

    def save_samples(self, filename="checkpoint.example"):
        filepath = str(MODEL_SAVE_PATH / filename)
        with open(filepath, 'wb') as f:
            pickle.dump(self.train_sample, f, -1)

    def load_samples(self, filename="checkpoint.example"):
        filepath = str(MODEL_SAVE_PATH / filename)
        with open(filepath, 'rb') as f:
            self.train_sample = pickle.load(f)

    def play_with_human(self, human_first=True, checkpoint_name="best.pt"):
        t = threading.Thread(target=self.wm_chess_gui.loop)
        t.start()

        libtorch_best = ChessNetWrapper()
        libtorch_best.load(checkpoint_name)
        mcts_best = MCTS(libtorch_best.predict, 1600 * 6)

        # create wm_chess game
        human_color = self.wm_chess_gui.get_human_color()
        state = Chess()

        players = ["alpha", None, "human"] if human_color == 1 else ["human", None, "alpha"]
        player_index = human_color if human_first else -human_color

        self.wm_chess_gui.reset_status()

        while True:
            player = players[player_index + 1]

            # select move
            if player == "alpha":
                prob = mcts_best.get_action_probability(state, True)
                best_move = int(np.argmax(np.array(list(prob))))
                self.wm_chess_gui.execute_move(player_index, INDEX_TO_MOVE_DICT[best_move])
            else:
                self.wm_chess_gui.set_is_human(True)
                # wait human action
                while self.wm_chess_gui.get_is_human():
                    time.sleep(0.1)
                best_move = self.wm_chess_gui.get_human_move()

            s = board_to_torch_state(state.get_board(), state.get_current_player())
            v, p = mcts_best.model_predict(s)
            print(f"before action, player = {state.get_current_player()}, v = {v}")

            s = board_to_torch_state(state.get_board(), - state.get_current_player())
            v, p = mcts_best.model_predict(s)
            print(f"before action, player = {-state.get_current_player()}, v = {v}")
            # execute move
            state.do_action(INDEX_TO_MOVE_DICT[best_move])
            s = board_to_torch_state(state.get_board(), state.get_current_player())
            v, p = mcts_best.model_predict(s)
            print(f"after action, player = {state.get_current_player()}, v = {v}")

            s = board_to_torch_state(state.get_board(), - state.get_current_player())
            v, p = mcts_best.model_predict(s)
            print(f"after action, player = {-state.get_current_player()}, v = {v}\n\n")

            # check game status
            ended, winner = state.is_end()
            if ended is True:
                win_string = "HUMAN WIN" if winner == human_color else "ALPHA ZERO WIN"
                self.wm_chess_gui.draw_end_string(win_string)
                break

            # update tree search
            mcts_best.update_tree(best_move)

            # next player
            player_index = -player_index

        print(win_string)


if __name__ == '__main__':
    temp = Trainer()
    temp.contest(2)
