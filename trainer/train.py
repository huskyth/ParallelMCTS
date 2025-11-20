import sys
from collections import deque
import swanlab
import numpy as np
import torch
from chess.common import ROOT_PATH, INDEX_TO_MOVE_DICT
from utils.math_tool import dirichlet_noise

from chess.chess import Chess
from chess.wm_chess_gui import WMChessGUI
from mcts.pure_mcts import MCTS
from models.network_wrapper import ChessNetWrapper
from utils.concurrent_tool import ConcurrentProcess


class Trainer:
    def __init__(self, train_config=None, use_swanlab=True, mode='train'):
        if use_swanlab:
            swanlab.login(api_key="rdGaOSnlBY0KBDnNdkzja")
            self.swanlab = swanlab.init(project="Chess", logdir=ROOT_PATH / "logs")
        else:
            self.swanlab = None
        self.train_config = train_config

        self.network = ChessNetWrapper()
        self.random_network = ChessNetWrapper()

        self.mcts = MCTS(self.network.predict, mode='train')

        self.state = Chess()

        self.train_sample = deque(maxlen=1000)

        self.best_win_rate = 0
        self.self_play_parallel_num = 5
        self.self_play_processor = ConcurrentProcess(self.self_play_parallel_num)
        self.contest_parallel_num = 5
        self.contest_processor = ConcurrentProcess(self.contest_parallel_num)

        if mode == 'play':
            epoch = self.network.load("best.pt")
            print(f"load {epoch} checkpoint tp play")
            self.pc_player = MCTS(self.network.predict, mode='test')
            self.wm_chess_gui = WMChessGUI(7, -1, is_show=True, mcts_player=self.step, play_state=self.state)
            self.wm_chess_gui.set_is_human(True)
            self.wm_chess_gui.start()
            self.state.reset(-1)
        elif mode == 'test':
            self.wm_chess_gui = None
            self.network.load("best.pt")
            self.random_network = ChessNetWrapper(None)
            result = self._contest()
            print(f"本次测试的报告为新模型胜率为{result[0] / sum(result)}")

    def _collect(self):

        param = [(self.mcts, self.state, i) for i in range(self.self_play_parallel_num)]
        self.self_play_processor.process(self._play, param)
        result = self.self_play_processor.result
        return [dim1 for dim2 in result for dim1 in dim2]

    @staticmethod
    def _play(mcts, state, i):
        print(f"😊 开始第{i}次自我Play")
        train_sample = []

        mcts.update_tree(-1)
        state.reset()

        while not state.is_end()[0]:
            probability = mcts.get_action_probability(state=state, is_greedy=False)
            ava_py_idx = [idx for idx, p in enumerate(probability) if p > 0]
            ava_py = [p for idx, p in enumerate(probability) if p > 0]
            ava_py_noise = dirichlet_noise(ava_py)
            action_idx = np.random.choice(len(ava_py_noise), p=ava_py_noise)
            action = ava_py_idx[action_idx]

            train_sample.append([state.get_torch_state(), probability, state.get_current_player()])

            state.do_action(action)
            mcts.update_tree(action)

        _, winner = state.is_end()
        assert winner is not None
        for item in train_sample:
            if item[-1] == winner:
                item.append(torch.tensor(1.0))
            else:
                item.append(torch.tensor(-1.0))
        return train_sample

    def _contest(self):
        param = []
        new_player = MCTS(self.network.predict, mode='test')
        old_mcts = MCTS(self.random_network.predict, mode='test')
        for i in range(self.contest_parallel_num):
            param.append((self.state, new_player, old_mcts, i))
        self.contest_processor.process(self._contest_one_time, param)
        ret = self.contest_processor.result
        new_win, old_win, draws = 0, 0, 0
        for item in ret:
            new_win_, old_win_, draws_, length_of_turn_ = item
            new_win += new_win_
            old_win += old_win_
            draws += draws_
            print(f"♬ 本局进行了{length_of_turn_}轮")

        return new_win, old_win, draws

    @staticmethod
    def _contest_one_time(state, new_player, old_mcts, i):

        new_win, old_win, draws = 0, 0, 0
        new_player.update_tree(-1)
        old_mcts.update_tree(-1)
        state.reset()
        player_list = [old_mcts, None, new_player]
        current_player = 1 if i % 2 == 0 else -1
        start_player = current_player
        print(f"🌟 start {i}th contest, first hand is {start_player}")
        length_of_turn = 0
        while not state.is_end()[0]:
            length_of_turn += 1
            player = player_list[current_player + 1]
            probability_new = player.get_action_probability(state, True)
            max_act = np.argmax(probability_new).item()
            state.do_action(max_act)
            old_mcts.update_tree(-1)
            new_player.update_tree(-1)
            current_player *= -1
        _, winner = state.is_end()
        if winner == 1:
            if start_player == 1:
                new_win += 1
            else:
                old_win += 1
        elif winner == -1:
            if start_player == 1:
                old_win += 1
            else:
                new_win += 1

        print(f"🍑 draws is {draws}, old win is {old_win}, new win is {new_win}")
        return new_win, old_win, draws, length_of_turn

    def _try_load(self):
        try:
            epoch = self.network.load("latest.pt")
        except Exception as e:
            print(e)
            epoch = 0

        return epoch

    def learn(self):
        start_epoch = self._try_load()

        for epoch in range(start_epoch, self.train_config.epoch):

            train_sample = self._collect()

            self.train_sample.extend(train_sample)

            if len(self.train_sample) >= 100:
                print(f"start training... size of train_sample: {len(self.train_sample)}")
                np.random.shuffle(self.train_sample)
                stat = self.network.train(self.train_sample)
                self.network.save(epoch)
                self.swanlab.log(stat)

            if (epoch + 1) % self.train_config.test_rate == 0:
                new_win, old_win, draws = self._contest()
                all_ = new_win + old_win + draws
                self.swanlab.log({
                    "win_new": new_win, "win_random": old_win, "draws": draws
                })
                if new_win / all_ > self.best_win_rate:
                    print(f"🍤 ACCEPT, Win Rate {new_win / all_} model saved")
                    self.network.save(epoch, key="best.pt")
                else:
                    print("🐑 REJECT")

    def step(self):
        self.pc_player.update_tree(-1)
        probability_new = self.pc_player.get_action_probability(self.state, True)
        max_act = np.argmax(probability_new).item()
        self.wm_chess_gui.execute_move(self.state.get_current_player(), INDEX_TO_MOVE_DICT[max_act])
        self.state.do_action(max_act)


if __name__ == '__main__':
    # temp = []
    # s = Chess().get_torch_state()
    # p = np.array([0] * 72)
    # for i in range(60):
    #     temp.append([s, p, 1, torch.tensor(1)])
    # dp = deque(maxlen=1000)
    # dp.extend(temp)
    # ChessNetWrapper(None).train(dp)

    temp = ChessNetWrapper(None)
    temp.load("best.pt")
    print(md)
