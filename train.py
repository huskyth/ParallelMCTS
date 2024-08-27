import os
import random
import sys
import threading
import time
from collections import deque
from functools import reduce

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
    WM_CHESS_GUI = WMChessGUI(7, -1)

    def __init__(self, is_eval=False):
        self.epoch = 10000
        self.test_rate = 20
        self.greedy_times = 5
        self.dirichlet_rate = 0.1
        self.dirichlet_probability = 0.3
        self.contest_number = 8
        self.self_play_number = 8
        self.batch_size = 512
        self.current_network = ChessNetWrapper()
        self.best_network = ChessNetWrapper()
        self.state = Chess()
        self.train_sample = deque([], maxlen=20)
        self.writer = MySummary(use_wandb=not is_eval)
        self.new_player = MCTS(self.current_network.predict)
        self.old_mcts = MCTS(self.current_network.predict)

    def _load(self):
        multiprocessing.set_start_method("spawn")
        print(f"process start method {multiprocessing.get_start_method()}")
        if os.path.exists(str(MODEL_SAVE_PATH / "checkpoint.pt")) and os.path.exists(
                str(MODEL_SAVE_PATH / "checkpoint.example")):
            print("load from old pth...")
            self.current_network.load()
            self.load_samples()
        else:
            self.current_network.save("best_checkpoint.pt")

    def _collect(self):
        temp = []
        with ProcessPoolExecutor(max_workers=os.cpu_count()) as ppe:
            future_list = [
                ppe.submit(Trainer._play, 1 if i % 2 == 0 else -1, self.current_network, i == 0) for i in
                range(self.self_play_number)]
            for k, item in enumerate(future_list):
                data = item.result()
                temp += data
        print(f"return temp length {len(temp)}")
        return temp

    @staticmethod
    def get_symmetries(board, pi, last_action, current_player):
        ret = [(board, pi, last_action, current_player, "origin")]
        new_board, new_last_action, new_pi, new_current_player = \
            lr(board, last_action, pi, current_player)
        ret.append((new_board, new_pi, new_last_action, new_current_player, "lr"))

        new_board, new_last_action, new_pi, new_current_player = \
            tb_(board, last_action, pi, current_player)
        ret.append((new_board, new_pi, new_last_action, new_current_player, "tb"))

        new_board_1, new_last_action_1, new_pi_1, new_current_player_1 = \
            lr(new_board, new_last_action, new_pi, new_current_player)
        ret.append((new_board_1, new_pi_1, new_last_action_1, new_current_player_1, "center"))
        return ret

    @staticmethod
    def _play(current_player, network, show):
        if show:
            t = threading.Thread(target=Trainer.WM_CHESS_GUI.loop)
            t.daemon = True
            t.start()
        train_sample = []
        player_1 = MCTS(network.predict)
        player_2 = MCTS(network.predict)
        player_list = [player_2, None, player_1]

        state = Chess()
        state.reset(current_player)

        if show:
            Trainer.WM_CHESS_GUI.reset_status()
        play_index = 1
        step = 0
        while not state.is_end()[0]:
            step += 1
            player = player_list[play_index + 1]

            is_greedy = step > 5
            probability = player.get_action_probability(state=state, is_greedy=is_greedy)
            last_action = state.last_action
            temp = Trainer.get_symmetries(state.get_board(), probability, last_action,
                                          state.get_current_player())
            # TODO://如有问题，一起测试这里
            for board, pi, last_action, current_player, _ in temp:
                board = board_to_torch_state(board, current_player, last_action)
                train_sample.append([board, pi, current_player])

            legal_action = state.get_legal_moves(state.get_current_player())
            legal_action = [MOVE_TO_INDEX_DICT[x] for x in legal_action]

            dirichlet_noise = 0.1 * np.random.dirichlet(0.3 * np.ones(len(legal_action)))

            probability = 0.9 * probability
            for i in range(len(legal_action)):
                probability[legal_action[i]] += dirichlet_noise[i]
            probability = probability / probability.sum()
            action = np.random.choice(len(probability), p=probability)

            if show:
                Trainer.WM_CHESS_GUI.execute_move(state.get_current_player(), INDEX_TO_MOVE_DICT[action])
            state.do_action(action)
            player_1.update_tree(action)
            player_2.update_tree(action)
            play_index *= -1
        print(f"{current_player} ended")
        _, winner = state.is_end()
        assert winner is not None
        for item in train_sample:
            if item[-1] == winner:
                item.append(1)
            else:
                item.append(-1)
        if show:
            Trainer.WM_CHESS_GUI.stop()
            t.join()
        return train_sample

    def contest(self):
        self.best_network.load(key="best_checkpoint.pt")
        new_win, old_win, draws = 0, 0, 0
        with ProcessPoolExecutor(max_workers=os.cpu_count()) as ppe:
            future_list = [
                ppe.submit(Trainer._contest, self.current_network, self.best_network, 1 if i % 2 == 0 else -1, i == 0)
                for i in range(self.contest_number)]
            for item in as_completed(future_list):
                winner = item.result()
                if winner == 1:
                    new_win += 1
                elif winner == -1:
                    old_win += 1
                else:
                    draws += 1
        print(f"return contest")
        return new_win, old_win, draws

    @staticmethod
    def _contest(network1, network2, current_player, show):
        if show:
            t = threading.Thread(target=Trainer.WM_CHESS_GUI.loop)
            t.daemon = True
            t.start()
        player1 = MCTS(network1.predict)
        player2 = MCTS(network2.predict)
        player_list = [player2, None, player1]
        state = Chess()
        state.reset(current_player)
        play_index = current_player

        if show:
            Trainer.WM_CHESS_GUI.reset_status()
        step = 0
        while not state.is_end()[0]:
            step += 1
            player = player_list[play_index + 1]
            probability_new = player.get_action_probability(state, True)
            max_act = int(np.argmax(probability_new))
            if show:
                Trainer.WM_CHESS_GUI.execute_move(state.get_current_player(), INDEX_TO_MOVE_DICT[max_act])
            state.do_action(max_act)

            player1.update_tree(max_act)
            player2.update_tree(max_act)
            play_index *= -1
            if step >= 450:
                return 0, 0, 1

        del network1
        del network2

        _, winner = state.is_end()
        if show:
            Trainer.WM_CHESS_GUI.stop()
            t.join()
        return winner

    def learn(self):
        self._load()
        for epoch in range(self.epoch):
            self.writer.add_float(epoch, "Epoch")

            train_sample = self._collect()
            self.train_sample.append(train_sample)
            train_data = reduce(lambda a, b: a + b, self.train_sample)
            random.shuffle(train_data)

            epoch_numbers = 1.5 * (len(train_sample) + self.batch_size - 1) // self.batch_size
            self.current_network.train(train_data, self.writer, int(epoch_numbers), self.batch_size)
            self.current_network.save()
            self.save_samples()

            if (epoch + 1) % self.test_rate == 0:
                new_win, old_win, draws = self.contest()

                self.writer.add_float(y=new_win, title="New player winning number")
                self.writer.add_float(y=old_win, title="Old player winning number")
                self.writer.add_float(y=draws, title="Draws number")

                if (new_win + old_win) > 0 and new_win / (new_win + old_win) > 0.55:
                    win_rate = float(new_win) / (new_win + old_win)
                    print("ACCEPT")
                    self.current_network.save("best_checkpoint.pt")
                else:
                    win_rate = -1 if new_win + old_win == 0 else new_win / (new_win + old_win)
                    print("REJECT")
                self.writer.add_float(y=win_rate, title="Winning rate")

    def save_samples(self, filename="checkpoint.example"):
        filepath = str(MODEL_SAVE_PATH / filename)
        with open(filepath, 'wb') as f:
            pickle.dump(self.train_sample, f, -1)

    def load_samples(self, filename="checkpoint.example"):
        filepath = str(MODEL_SAVE_PATH / filename)
        with open(filepath, 'rb') as f:
            self.train_sample = pickle.load(f)

    def play_with_human(self, human_first=True, checkpoint_name="checkpoint.pt"):
        t = threading.Thread(target=Trainer.WM_CHESS_GUI.loop)
        t.start()

        libtorch_best = ChessNetWrapper()
        libtorch_best.load(checkpoint_name)
        mcts_best = MCTS(libtorch_best.predict, 1600 * 6)

        # create wm_chess game
        human_color = Trainer.WM_CHESS_GUI.get_human_color()
        state = Chess()

        players = ["alpha", None, "human"] if human_color == 1 else ["human", None, "alpha"]
        player_index = human_color if human_first else -human_color

        state.reset(player_index)

        Trainer.WM_CHESS_GUI.reset_status()

        while True:
            player = players[player_index + 1]

            # select move
            if player == "alpha":
                prob = mcts_best.get_action_probability(state, True)
                best_move = int(np.argmax(np.array(list(prob))))
                Trainer.WM_CHESS_GUI.execute_move(player_index, INDEX_TO_MOVE_DICT[best_move])
            else:
                Trainer.WM_CHESS_GUI.set_is_human(True)
                # wait human action
                while Trainer.WM_CHESS_GUI.get_is_human():
                    time.sleep(0.1)
                best_move = Trainer.WM_CHESS_GUI.get_human_move()

            last_action = state.last_action
            s = board_to_torch_state(state.get_board(), state.get_current_player(), last_action)
            v, p = mcts_best.model_predict(s)
            # TODO://To Check get_torch_state and board_to_torch_state is equal
            print(f"before action, player = {state.get_current_player()}, v = {v}, last_action = {last_action}")

            s = board_to_torch_state(state.get_board(), - state.get_current_player(), last_action)
            v, p = mcts_best.model_predict(s)
            print(f"before action, player = {-state.get_current_player()}, v = {v}, last_action = {last_action}")
            # execute move
            state.do_action(INDEX_TO_MOVE_DICT[best_move])

            last_action = state.last_action
            s = board_to_torch_state(state.get_board(), state.get_current_player(), last_action)
            v, p = mcts_best.model_predict(s)
            print(f"after action, player = {state.get_current_player()}, v = {v}, last_action = {last_action}")

            s = board_to_torch_state(state.get_board(), - state.get_current_player(), last_action)
            v, p = mcts_best.model_predict(s)
            print(f"after action, player = {-state.get_current_player()}, v = {v}, last_action = {last_action}\n\n")

            # check game status
            ended, winner = state.is_end()
            if ended is True:
                win_string = "HUMAN WIN" if winner == human_color else "ALPHA ZERO WIN"
                Trainer.WM_CHESS_GUI.draw_end_string(win_string)
                break

            # update tree search
            mcts_best.update_tree(best_move)

            # next player
            player_index = -player_index

        print(win_string)


if __name__ == '__main__':
    temp = Trainer()
    temp.contest(2)
