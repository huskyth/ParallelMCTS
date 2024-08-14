import sys
import threading
from collections import deque

import numpy as np

from chess.common import ROOT_PATH, INDEX_TO_MOVE_DICT
from tensor_board_tool import MySummary

path = str(ROOT_PATH / "chess")
if path not in sys.path:
    sys.path.append(path)
else:
    print("path already in")
from chess.chess import Chess
from chess.wm_chess_gui import WMChessGUI
from mcts.pure_mcts import MCTS
from network_wrapper import ChessNetWrapper


class Trainer:
    def __init__(self):
        self.epoch = 100
        self.test_rate = 5
        self.greedy_times = 5
        self.dirichlet_rate = 1 - 0.25
        self.dirichlet_probability = 0.3
        self.use_gui = True
        self.network = ChessNetWrapper()
        self.old_network = ChessNetWrapper()
        self.mcts = MCTS(self.network.predict)
        self.state = Chess()
        self.train_sample = deque(maxlen=1000)
        self.wm_chess_gui = WMChessGUI(7, -1)
        self.writer = MySummary(use_wandb=True)

    def _collect(self):
        return self._play()

    def _play(self):
        step = 0
        train_sample = []
        self.mcts.update_tree(-1)
        self.state.reset()
        if self.use_gui:
            self.wm_chess_gui.reset_status()
        while not self.state.is_end()[0]:
            step += 1
            is_greedy = step < self.greedy_times
            probability = self.mcts.get_action_probability(state=self.state, is_greedy=is_greedy)
            action = np.random.choice(len(probability), p=probability)
            if self.use_gui:
                self.wm_chess_gui.execute_move(self.state.get_current_player(), INDEX_TO_MOVE_DICT[action])
            train_sample.append([self.state.get_torch_state(), probability, self.state.get_current_player()])
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

    def _contest(self, n):
        new_player = MCTS(self.network.predict)
        self.old_network.load("old_version.pt")
        old_mcts = MCTS(self.old_network.predict)
        new_win, old_win, draws = 0, 0, 0
        for i in range(n):
            new_player.update_tree(-1)
            old_mcts.update_tree(-1)
            self.state.reset()
            player_list = [old_mcts, None, new_player]
            step = 0
            current_player = 1
            while not self.state.is_end()[0]:
                step += 1
                player = player_list[current_player + 1]
                probability_new = player.get_action_probability(self.state, True)
                max_act = np.argmax(probability_new).item()
                self.state.do_action(max_act)
                new_player.update_tree(max_act)
                old_mcts.update_tree(max_act)
                current_player *= -1
            self.writer.add_float(y=step, title="Testing episode length")
            _, winner = self.state.is_end()
            assert winner is not None
            if winner == 1:
                new_win += 1
            else:
                old_win += 1
        draws = n - new_win - old_win
        self.writer.add_float(y=new_win, title="New player winning number")
        self.writer.add_float(y=old_win, title="Old player winning number")
        self.writer.add_float(y=new_win / n, title="Winning rate")
        return new_win, old_win, draws

    def learn(self):
        if self.use_gui:
            t = threading.Thread(target=self.wm_chess_gui.loop)
            t.start()
        for epoch in range(self.epoch):
            train_sample = self._collect()
            self.train_sample.append(train_sample)
            self.network.save("old_version.pt")
            if len(self.train_sample) >= 10:
                np.random.shuffle(self.train_sample)
                self.network.train(self.train_sample)

            if (epoch + 1) % self.test_rate == 0:
                new_win, old_win, draws = self._contest(10)
                all_ = new_win + old_win + draws

                if new_win / all_ > 0.6:
                    print("ACCEPT")
                    self.network.save("best.pt")
                else:
                    print("REJECT")


if __name__ == '__main__':
    temp = [0.1, 0.2, 0.3, 0.4]
    temp = np.array(temp)
    y = np.random.choice(len(temp), p=temp)
    print(y)
