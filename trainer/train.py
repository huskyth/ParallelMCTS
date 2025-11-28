import random
from collections import deque
import swanlab
import numpy as np
import torch
from constants import ROOT_PATH

from utils.concurrent_tool import ConcurrentProcess


class Trainer:
    def __init__(self, train_config=None, use_swanlab=True, mode='train', number_of_self_play=5, number_of_contest=5,
                 abstract_game=None, use_pool=False, is_render=False):
        if use_swanlab:
            swanlab.login(api_key="rdGaOSnlBY0KBDnNdkzja")
            self.swanlab = swanlab.init(project="Chess", logdir=ROOT_PATH / "logs")
        else:
            self.swanlab = None
        self.train_config = train_config

        self.abstract_game = abstract_game

        self.train_sample = deque(maxlen=1000)
        self.is_render = is_render
        self.best_win_rate = 0
        self.use_pool = use_pool
        if use_pool:
            self.self_play_parallel_num = number_of_self_play
            self.self_play_processor = ConcurrentProcess(self.self_play_parallel_num)
            self.contest_parallel_num = number_of_contest
            self.contest_processor = ConcurrentProcess(self.contest_parallel_num)
        else:
            self.self_play_num = number_of_self_play
            self.contest_num = number_of_contest

    def _collect_concurrent(self):
        mcts = self.abstract_game.mcts
        state = self.abstract_game.state
        param = [(mcts, state, i) for i in range(self.self_play_parallel_num)]
        self.self_play_processor.process(self._self_play, param)
        result = self.self_play_processor.result
        return [dim1 for dim2 in result for dim1 in dim2]

    def _collect(self):
        sample = []
        mcts = self.abstract_game.mcts
        state = self.abstract_game.state
        for i in range(self.self_play_num):
            temp = self._self_play(mcts, state, i)
            sample.extend(temp)
        return sample

    @staticmethod
    def _self_play(mcts, state, i):
        print(f"😊 开始第{i + 1}次自我Play")
        train_sample = []

        mcts.update_tree(-1)
        state.reset()

        while not state.is_end()[0]:
            probability = mcts.get_action_probability(state=state, is_greedy=False)
            action = np.random.choice(len(probability), p=probability)

            train_sample.append([state.get_torch_state(), probability, state.get_current_player()])

            state.do_action(action)
            mcts.update_tree(action)

        _, winner = state.is_end()
        assert winner is not None
        for item in train_sample:
            if winner == 0:
                item.append(torch.tensor(0.0))
            elif item[-1] == winner:
                item.append(torch.tensor(1.0))
            else:
                item.append(torch.tensor(-1.0))

        return train_sample

    def _contest_concurrent(self):
        param = []
        new_player = self.abstract_game.mcts
        old_mcts = self.abstract_game.random_mcts
        state = self.abstract_game.state
        for i in range(self.contest_parallel_num):
            param.append((state, new_player, old_mcts, i))
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

    def _contest(self):
        new_player = self.abstract_game.mcts
        state = self.abstract_game.state
        wins = 0
        olds = 0
        draws = 0
        for i in range(self.contest_num):
            new_win, old_win, draw, length_of_turn = self._contest_one_time(state, new_player, i)
            print(f"♬ 本局进行了{length_of_turn}轮\n")
            wins += new_win
            olds += old_win
            draws += draw
        return wins, olds, draws

    @staticmethod
    def _contest_one_time(state, new_player, i):

        new_win, old_win, draws = 0, 0, 0
        new_player.update_tree(-1)
        state.reset()
        player_list = [None, None, new_player]
        current_player = 1 if i % 2 == 0 else -1
        start_player = current_player
        print(f"\n🌟 start {i}th contest, first hand is {start_player}")
        length_of_turn = 0
        state.render("初始化局面")
        while not state.is_end()[0]:
            length_of_turn += 1
            player = player_list[current_player + 1]
            if player is None:
                tuple_act = random.choice(state.get_legal_moves(state.get_current_player()))
                max_act = tuple_act[0] * 3 + tuple_act[1]
            else:
                probability_new = player.get_action_probability(state, True)
                max_act = np.argmax(probability_new).item()

            state.render(
                f"当前玩家 {player.name if player else '随机玩家'} {state.get_current_player()}, 执行 {max_act}")
            state.do_action(max_act)
            state.render(f"当前玩家 {-state.get_current_player()}")
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
        elif winner == 0:
            draws += 1
        if winner == 1:
            win = 'X'
        elif winner == -1:
            win = 'O'
        else:
            win = '平局'
        state.render(f"本局-{i} 游戏结束,{win} 获胜")

        print(f"🍑 draws is {draws}, old win is {old_win}, new win is {new_win}")
        return new_win, old_win, draws, length_of_turn

    def test(self):
        self.abstract_game.network.load("best.pt")
        self.abstract_game.network.eval()
        if self.use_pool:
            new_win, old_win, draws = self._contest_concurrent()
        else:
            new_win, old_win, draws = self._contest()
        all_ = new_win + old_win + draws
        self.swanlab.log({
            "win_new": new_win, "win_random": old_win, "draws": draws, "win_rate": new_win / all_
        })
        print(f"🍤 Win Rate {new_win / all_}")

    def play(self, current_player="AI"):
        if current_player not in ["AI", "Human"]:
            raise ValueError("current_player must be 'AI' or 'Human'")
        self.abstract_game.network.load("best.pt")
        ai = self.abstract_game.mcts
        state = self.abstract_game.state

        ai.update_tree(-1)
        state.reset()
        start_player = current_player
        ano_player = 'Human' if start_player == 'AI' else 'AI'
        state.is_render = True
        state.render("当前局面")
        state.is_render = False
        while not state.is_end()[0]:
            if current_player == "AI":
                print('👀 Now AI play')
                probability_new = ai.get_action_probability(state, True)
                max_act = np.argmax(probability_new).item()
                current_player = "Human"
            else:
                print('👀 Now human play')
                max_act = int(input("please input you action"))
                current_player = "AI"

            state.do_action(max_act)
            ai.update_tree(-1)
            state.is_render = True
            state.render("当前局面")
            state.is_render = False
        _, winner = state.is_end()
        if winner == 0:
            print("和棋")
        elif winner == 1:
            print(f"{start_player} 赢了")
        elif winner == -1:
            print(f"{ano_player} 赢了")

    def learn(self):
        start_epoch = self.abstract_game.start_epoch
        is_trained = False
        for epoch in range(start_epoch, self.train_config.epoch):

            if self.use_pool:
                train_sample = self._collect_concurrent()
            else:
                train_sample = self._collect()

            self.train_sample.extend(train_sample)

            if len(self.train_sample) >= 50:
                is_trained = True
                print(f"start training... size of train_sample: {len(self.train_sample)}")
                np.random.shuffle(self.train_sample)
                self.abstract_game.network.save(epoch, key="before_train.pt")
                stat = self.abstract_game.network.train_net(self.train_sample)
                self.abstract_game.network.save(epoch)
                for sta in stat:
                    self.swanlab.log(sta)

            if (epoch + 1) % self.train_config.test_rate == 0 and is_trained:
                self.abstract_game.network.eval()
                if self.use_pool:
                    new_win, old_win, draws = self._contest_concurrent()
                else:
                    new_win, old_win, draws = self._contest()
                all_ = new_win + old_win + draws
                self.swanlab.log({
                    "win_new": new_win, "win_random": old_win, "draws": draws, "win_rate": new_win / all_
                })
                if new_win / all_ > self.best_win_rate:
                    print(f"🍤 ACCEPT, Win Rate {new_win / all_} model saved large than {self.best_win_rate}")
                    self.best_win_rate = new_win / all_
                    self.abstract_game.network.save(epoch, key="best.pt")
                else:
                    print("🐑 REJECT")
                    # self.abstract_game.network.load(key="before_train.pt")
