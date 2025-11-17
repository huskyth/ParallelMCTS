import sys
from collections import deque
import swanlab
import numpy as np
import torch
from chess.common import ROOT_PATH, INDEX_TO_MOVE_DICT

path = str(ROOT_PATH / "chess")
if path not in sys.path:
    sys.path.append(path)
else:
    print("path already in")
from chess.chess import Chess
from chess.wm_chess_gui import WMChessGUI
from mcts.pure_mcts import MCTS
from models.network_wrapper import ChessNetWrapper


class Trainer:
    def __init__(self, use_gui=False, train_config=None):
        swanlab.login(api_key="rdGaOSnlBY0KBDnNdkzja")
        self.swanlab = swanlab.init(project="Chess", logdir=ROOT_PATH / "logs")
        self.train_config = train_config

        self.network = ChessNetWrapper(self.swanlab)
        self.random_network = ChessNetWrapper(self.swanlab)

        self.mcts = MCTS(self.network.predict)

        self.state = Chess()

        self.train_sample = deque(maxlen=1000)
        self.wm_chess_gui = WMChessGUI(7, -1, is_show=use_gui)

        self.best_win_rate = 0

    def _collect(self):
        return self._play()

    def _play(self):
        train_sample = []

        self.mcts.update_tree(-1)

        self.state.reset()

        self.wm_chess_gui.reset_status()

        while not self.state.is_end()[0]:
            probability = self.mcts.get_action_probability(state=self.state, is_greedy=False)

            action = np.random.choice(len(probability), p=probability)

            self.wm_chess_gui.execute_move(self.state.get_current_player(), INDEX_TO_MOVE_DICT[action])

            train_sample.append([self.state.get_torch_state(), probability, self.state.get_current_player()])

            self.state.do_action(action)
            self.mcts.update_tree(action)

        _, winner = self.state.is_end()
        assert winner is not None
        for item in train_sample:
            if item[-1] == winner:
                item.append(torch.tensor(1.0))
            else:
                item.append(torch.tensor(-1.0))
        return train_sample

    def _contest(self, n):

        new_player = MCTS(self.network.predict)
        old_mcts = MCTS(self.random_network.predict)

        new_win, old_win, draws = 0, 0, 0
        for i in range(n):
            new_player.update_tree(-1)
            old_mcts.update_tree(-1)
            self.state.reset()
            player_list = [old_mcts, None, new_player]
            current_player = 1 if i % 2 == 0 else -1
            start_player = current_player
            while not self.state.is_end()[0]:
                player = player_list[current_player + 1]
                probability_new = player.get_action_probability(self.state, True)
                max_act = np.argmax(probability_new).item()
                self.state.do_action(max_act)

                new_player.update_tree(max_act)
                old_mcts.update_tree(max_act)
                current_player *= -1
            _, winner = self.state.is_end()
            assert winner is not None
            if winner == 1:
                if start_player == 1:
                    new_win += 1
                else:
                    old_win += 1
            else:
                if start_player == 1:
                    old_win += 1
                else:
                    new_win += 1
        draws = n - new_win - old_win
        return new_win, old_win, draws

    def _try_load(self):
        try:
            epoch = self.network.load("latest.pt")
        except Exception as e:
            print(e)
            epoch = 0

        return epoch

    def learn(self):
        start_epoch = self._try_load()
        self.wm_chess_gui.start()

        for epoch in range(start_epoch, self.train_config.epoch):

            train_sample = self._collect()

            self.train_sample.extend(train_sample)

            if len(self.train_sample) >= 10:
                print(f"start training... size of train_sample: {len(self.train_sample)}")
                np.random.shuffle(self.train_sample)
                self.network.train(self.train_sample)
                self.network.save(epoch)

            if (epoch + 1) % self.train_config.test_rate == 0:
                new_win, old_win, draws = self._contest(2)
                all_ = new_win + old_win + draws
                self.swanlab.log({
                    "win_new": new_win, "win_random": old_win, "draws": draws
                })
                if new_win / all_ > self.best_win_rate:
                    print(f"🍤 ACCEPT, {new_win / all_} model saved")
                    self.network.save(epoch, key="best.pt")
                else:
                    print("🐑 REJECT")


if __name__ == '__main__':
    temp = []
    s = Chess().get_torch_state()
    p = np.array([0] * 72)
    for i in range(60):
        temp.append([s, p, 1, torch.tensor(1)])
    dp = deque(maxlen=1000)
    dp.extend(temp)
    ChessNetWrapper(None).train(dp)