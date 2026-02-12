import os
from collections import deque
import swanlab
import numpy as np
import torch
from constants import ROOT_PATH
from game.chess.chess import Chess
from game.tictactoe.tictactoe import TicTacToe
from mcts.pure_mcts import MCTS
from models.tictactoe.network_wrapper import TictactoeNetWrapper
from models.wm_model.network_wrapper import ChessNetWrapper
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from pickle import Pickler, Unpickler

from utils.math_tool import dirichlet_noise


class Trainer:
    def __init__(self, train_config=None, use_swanlab=True, mode='train', number_of_self_play=5, number_of_contest=5,
                 use_pool=False, is_render=False, is_data_augment=False, is_image_show=False, is_continue=False,
                 game="WMChess"):

        self.is_continue = is_continue
        if use_swanlab:
            swanlab.login(api_key="rdGaOSnlBY0KBDnNdkzja")
            self.swanlab = swanlab.init(project="ChessGame", logdir=ROOT_PATH / "logs")
        else:
            self.swanlab = None
        self.train_config = train_config
        self.init_best_model = False

        self.is_data_augment = is_data_augment
        self.is_image_show = is_image_show

        self.train_sample = []
        self.is_render = is_render
        self.use_pool = use_pool
        self.current_play_turn = 0
        self.self_play_num = number_of_self_play

        if use_pool:
            self.self_play_parallel_num = number_of_self_play
            self.contest_parallel_num = number_of_contest
        else:
            self.contest_num = number_of_contest
        self.game = game
        self.training_network = self.generate_net(self.game)

    @staticmethod
    def generate_state(game):
        state = None
        if game == 'tictactoe':
            state = TicTacToe()
        elif game == 'WMChess':
            state = Chess()
        return state

    @staticmethod
    def generate_net(game):
        net = None
        if game == "tictactoe":
            net = TictactoeNetWrapper()
        elif game == 'WMChess':
            net = ChessNetWrapper()
        return net

    def _collect_concurrent(self):

        state = self.generate_state(self.game)
        param = (state, False, self.is_data_augment, self.is_image_show, self.game)

        result = []
        with ProcessPoolExecutor(max_workers=min(4, os.cpu_count())) as ppe:
            future_list = [ppe.submit(self.self_play_concurrent, i + self.current_play_turn, *param)
                           for i in range(self.self_play_parallel_num)]
            for item in as_completed(future_list):
                data_list = item.result()
                result.extend(data_list)

        self.current_play_turn += self.self_play_parallel_num
        return result

    def _collect(self):
        sample = deque([], maxlen=200000)
        mcts1 = MCTS(self.training_network.predict, mode='train', name="自我对弈玩家1")
        mcts2 = MCTS(self.training_network.predict, mode='train', name="自我对弈玩家2")
        state = self.generate_state(self.game)
        for _ in tqdm(range(self.self_play_num), desc='Self Play'):
            temp = self._self_play(self.current_play_turn, mcts1, mcts2, state, self.is_render, self.is_data_augment,
                                   self.is_image_show)
            self.current_play_turn += 1
            sample.extend(temp)
        return sample

    @staticmethod
    def self_play_concurrent(current_play_turn, state, is_render, is_data_augment, is_image_show, game):
        net = Trainer.generate_net(game)
        net.load("best.pt")
        mcts1 = MCTS(net.predict, mode='train', name="自我对弈玩家1")
        mcts2 = MCTS(net.predict, mode='train', name="自我对弈玩家2")
        return Trainer._self_play(current_play_turn, mcts1, mcts2, state, is_render, is_data_augment, is_image_show)

    @staticmethod
    def _self_play(current_play_turn, mcts1, mcts2, state, is_render, is_data_augment, is_image_show):
        train_sample = []
        turn = 0
        mcts1.update_tree(-1)
        mcts2.update_tree(-1)
        if (current_play_turn + 1) % 2 == 0:
            player_list = [mcts2, None, mcts1]
        else:
            player_list = [mcts1, None, mcts2]
        state.reset()
        start_player = 1
        print(
            f"😊 开始第{current_play_turn + 1}轮self_play"
            f"先手name是 {player_list[start_player + 1].name}，"
            f"进程ID {os.getpid()}")

        state.image_show(f"测试局面", is_image_show)
        while not state.is_end()[0]:
            turn += 1
            is_greedy = turn > 200
            if turn % 100 == 0:
                print(f"😊 第{current_play_turn + 1}次self_play 共进行 {turn} 轮")

            probability = player_list[start_player + 1].get_action_probability(state=state, is_greedy=False)

            ava_py_noise = dirichlet_noise(probability[probability > 0], epison=0.4, alpha=0.4)
            probability[probability > 0] = ava_py_noise

            action = np.random.choice(len(probability), p=probability)
            train_sample.append(
                [state.get_torch_state().cpu(), torch.tensor(probability), state.get_current_player(), action])
            if is_data_augment:
                s1, p1 = state.top_buttom(state.get_torch_state(), probability)
                s2, p2 = state.left_right(state.get_torch_state(), probability)
                s3, p3 = state.center(state.get_torch_state(), probability)
                train_sample.append([s1, p1, state.get_current_player(), action])
                train_sample.append([s2, p2, state.get_current_player(), action])
                train_sample.append([s3, p3, state.get_current_player(), action])
            state.do_action(action)
            start_player *= -1
            mcts1.update_tree(action)
            mcts2.update_tree(action)
            state.image_show(f"测试局面", is_image_show)

        episode_length = len(train_sample)
        gama = 1
        print(f'☃️ 一共 {turn}轮')
        _, winner = state.is_end()
        print(f'☃️ 一共 {turn}轮, 结果为 {winner}')
        assert winner is not None
        for idx, item in enumerate(train_sample):
            rate = gama ** (episode_length - 1 - idx)
            if winner == 0:
                item.append(torch.tensor(0.0))
            elif item[-2] == winner:
                item.append(torch.tensor(1.0 * rate))
            else:
                item.append(torch.tensor(-1.0 * rate))

        if is_render:
            title = ["原始", "上下", "左右", "中心对称"]
            print("=" * 150 + f"训练数据， 当前训练数据有 {len(train_sample)} 比")
            rate_ = 4 if is_data_augment else 1
            for idx, item in enumerate(train_sample):
                state, p, player, act, value = item

                print(
                    f"\n\n" + "*" * 100 + " " + str(idx % rate_) + f" {title[idx % rate_]}"
                                                                   f"当前状态为\n{state[:, :, 0]}\n {state[:, :, 1]}\n\n {state[:, :, 2]}"
                                                                   f"\n（改了，改成了单一概率，是随机采样后修改为只有一个1的概率，看到不要惊讶）概率为{p}\n当前玩家{player} value = {value} 执行 {act}（仅对第一组有效）,应该执行的行为（最大概率） {np.argmax(p)}"
                    + '\n' + f"#" * 150)
            print("=" * 150 + "训练数据")
        for idx in range(len(train_sample)):
            train_sample[idx] = train_sample[idx][:3] + [train_sample[idx][4]]
        return train_sample

    def _contest_concurrent(self):
        return self.test_concurrent(self.contest_parallel_num, self._contest_one_time_concurrent)

    def _contest(self, test_number=1000):
        first_player = MCTS(self.training_network.predict, mode='test', name="当前训练玩家")
        state = self.generate_state(self.game)

        contest_network = self.generate_net(self.game)
        contest_network.load("before_train.pt")
        contest_network.eval()
        second_player = MCTS(contest_network.predict, mode='test', name="之前最优玩家")

        first_win = 0
        second_win = 0
        draws = 0
        for _ in tqdm(range(test_number // 2)):
            win1, win2, draw, length_of_turn = self._contest_one_time(state, first_player, second_player,
                                                                      self.is_image_show)
            print(f"♬ 本局进行了{length_of_turn}轮\n")
            first_win += win1
            second_win += win2
            draws += draw

        for _ in tqdm(range(test_number // 2)):
            win1, win2, draw, length_of_turn = self._contest_one_time(state, second_player, first_player,
                                                                      self.is_image_show)
            print(f"♬ 本局进行了{length_of_turn}轮\n")
            first_win += win2
            second_win += win1
            draws += draw
        return first_win, second_win, draws

    @staticmethod
    def _contest_one_time(state, first_player, second_player, is_image_show):
        first_player.update_tree(-1)
        second_player.update_tree(-1)
        player_list = [second_player, None, first_player]
        current_player = 1
        state.reset()
        length_of_turn = 0
        max_turn = 1000
        state.render("初始化局面")
        state.image_show("Contest", is_image_show)
        while not state.is_end()[0]:
            length_of_turn += 1
            if length_of_turn % 100 == 0:
                print(f"🍑 当前步数为 {length_of_turn}")
            player = player_list[current_player + 1]
            if player is None:
                max_act = state.move_random()
            else:
                probability_new = player.get_action_probability(state, True)
                max_act = np.argmax(probability_new).item()
            p_name = player.name if player else '随机玩家'
            state.render(
                f"Step {length_of_turn} - 当前玩家 {p_name} {state.get_current_player()}, 执行 {state.index_to_move[max_act]}")
            state.do_action(max_act)
            state.render(f"Step {length_of_turn} - 当前玩家 {p_name} 索引{-state.get_current_player()}执行后的局面")
            first_player.update_tree(max_act)
            second_player.update_tree(max_act)
            current_player *= -1
            state.image_show("Contest", is_image_show)

        first_win, second_win, draws = 0, 0, 0
        _, winner = state.is_end()
        if winner == 1:
            first_win = 1
        elif winner == -1:
            second_win = 1
        elif winner == 0:
            draws = 1

        return first_win, second_win, draws, length_of_turn

    @staticmethod
    def _test_one_time_concurrent(state, first_start, is_image_show, game):
        first_net = Trainer.generate_net(game)
        first_net.load("best.pt")
        second_net = Trainer.generate_net(game)
        first_player = MCTS(first_net.predict, mode='test', name="玩家1")
        second_player = MCTS(second_net.predict, mode='test', name="玩家2")
        if first_start == 1:
            first_player, second_player = second_player, first_player

        return Trainer._contest_one_time(state, first_player, second_player, is_image_show)

    @staticmethod
    def _contest_one_time_concurrent(state, first_start, is_image_show, game):
        first_net = Trainer.generate_net(game)
        second_net = Trainer.generate_net(game)
        first_net.load("latest.pt")
        second_net.load("before_train.pt")
        first_player = MCTS(first_net.predict, mode='test', name="玩家1")
        second_player = MCTS(second_net.predict, mode='test', name="玩家2")
        if first_start == 1:
            first_player, second_player = second_player, first_player

        return Trainer._contest_one_time(state, first_player, second_player, is_image_show)

    def test_(self, test_number):
        state = self.generate_state(self.game)

        first_net = self.generate_net(self.game)
        # first_net.load("best.pt")

        second_net = self.generate_net(self.game)
        second_net.load("best.pt")

        first_win = 0
        second_win = 0
        draws = 0
        for _ in tqdm(range(test_number // 2)):
            first_net = self.generate_net(self.game)
            # first_net.load("best.pt")
            first_player = MCTS(first_net.predict, mode='test', name="玩家1")

            second_player = MCTS(second_net.predict, mode='test', name="玩家2")

            win1, win2, draw, length_of_turn = self._contest_one_time(state, first_player, second_player,
                                                                      self.is_image_show)
            print(f"♬ 本局进行了{length_of_turn}轮\n")
            first_win += win1
            second_win += win2
            draws += draw

        print(f"模型1先行：  模型1：{first_win}, 模型2：{second_win}, 平局：{draws}")
        after_first = 0
        after_second = 0
        after_draw = 0
        for _ in tqdm(range(test_number // 2)):
            first_net = self.generate_net(self.game)
            # first_net.load("best.pt")
            first_player = MCTS(first_net.predict, mode='test', name="玩家1")

            second_player = MCTS(second_net.predict, mode='test', name="玩家2")

            win1, win2, draw, length_of_turn = self._contest_one_time(state, second_player, first_player,
                                                                      self.is_image_show)
            print(f"♬ 本局进行了{length_of_turn}轮\n")
            after_first += win2
            after_second += win1
            after_draw += draw
            first_win += win2
            second_win += win1
            draws += draw

        print(f"模型2先行：  模型1：{after_first}, 模型2：{after_second}, 平局：{after_draw}")

        print(f"最终结果：  模型1：{first_win}, 模型2：{second_win}, 平局：{draws}")

    def test_concurrent(self, test_number, test_fun):

        state = self.generate_state(self.game)
        new_win = 0
        old_win = 0
        draws = 0
        with ProcessPoolExecutor(max_workers=min(4, os.cpu_count())) as ppe:
            future_list = []
            for i in range(test_number // 2):
                show = None
                if i == 0:
                    show = True
                param = (state, 0, show, self.game)
                future_list.append(ppe.submit(test_fun, *param))
            for item in as_completed(future_list):
                data_list = item.result()
                new_win_, old_win_, draws_, length_of_turn_ = data_list
                new_win += new_win_
                old_win += old_win_
                draws += draws_
                print(f"♬ 训练玩家先行 本局进行了{length_of_turn_}轮 new_win_ {new_win_}，old_win_ {old_win_}\n")
        print(f"♬ 训练玩家先行 中间结果 new_win {new_win}，old_win {old_win} draws {draws}\n")
        with ProcessPoolExecutor(max_workers=min(4, os.cpu_count())) as ppe:
            future_list = []
            for _ in range(test_number // 2):
                param = (state, 1, None, self.game)
                future_list.append(ppe.submit(test_fun, *param))
            for item in as_completed(future_list):
                data_list = item.result()
                new_win_, old_win_, draws_, length_of_turn_ = data_list
                new_win += old_win_
                old_win += new_win_
                draws += draws_
                print(f"♬ 之前玩家先行 本局进行了{length_of_turn_}轮，new_win_ {new_win_}，old_win_ {old_win_}\n")
        print(f"♬ 之前玩家先行 最终结果 new_win {new_win}，old_win {old_win} draws {draws}\n")
        return new_win, old_win, draws

    def test(self, test_number):
        if self.use_pool:
            self.test_concurrent(test_number, self._test_one_time_concurrent)
        else:
            self.test_(test_number)

    def play(self, current_player="AI"):
        if current_player not in ["AI", "Human"]:
            raise ValueError("current_player must be 'AI' or 'Human'")
        if self.game == 'WMChess':
            self._wm_play()
            return
        elif self.game == 'tictactoe':
            state = TicTacToe(is_render=True)
            net = TictactoeNetWrapper()
            net.load("best.pt")
            player = MCTS(net.predict, mode='test', name="玩家")
            player.update_tree(-1)
            state.reset()
            start_player = current_player
            ano_player = 'Human' if start_player == 'AI' else 'AI'
            state.is_render = True
            state.render("当前局面")
            state.is_render = False
            while not state.is_end()[0]:
                value, probability = net.predict(state.get_torch_state())
                print(f"局面 概率 {probability}, {np.argmax(probability)}")
                if current_player == "AI":
                    print('👀 Now AI play')
                    probability_new = player.get_action_probability(state, True)
                    print(f"mcts策略 {probability_new}")
                    max_act = np.argmax(probability_new).item()
                    current_player = "Human"
                else:
                    print('👀 Now human play')
                    max_act = int(input("please input you action"))
                    current_player = "AI"

                state.do_action(max_act)
                player.update_tree(-1)
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

    def _wm_play(self):
        from game.chess.wm_chess_gui import WMChessGUI
        state = Chess(is_render=self.is_render)
        self.training_network.load("best.pt")
        self.training_network.eval()

        mcts = MCTS(self.training_network.predict, mode='test', name="AI", simulate_times=800)

        state.reset()
        wm = WMChessGUI(mcts, state)
        wm.start()

    def save_history(self):
        with open(self.training_network.MODEL_SAVE_PATH / "train_history.examples", "wb+") as f:
            Pickler(f).dump(self.train_sample)

    def load_history(self):
        with open(self.training_network.MODEL_SAVE_PATH / "train_history.examples", "rb") as f:
            self.train_sample = Unpickler(f).load()

    def learn(self):
        start_epoch = 0

        if self.is_continue:
            start_epoch = self.training_network.try_load()
            self.load_history()
        for epoch in range(start_epoch, self.train_config.epoch):

            train_sample = self._collect()

            self.train_sample.append(train_sample)
            if len(self.train_sample) > 20:
                self.train_sample.pop(0)

            self.save_history()
            train_sample = []
            for x in self.train_sample:
                train_sample.extend(x)
            print(f"start training... size of train_sample: {len(train_sample)}")
            np.random.shuffle(train_sample)
            self.training_network.save(epoch, key="before_train.pt")
            stat = self.training_network.train_net(train_sample, self.swanlab)
            self.training_network.save(epoch)
            # for sta in stat:
            #     self.swanlab.log(sta)

            self.training_network.eval()
            if self.use_pool:
                new_win, old_win, draws = self._contest_concurrent()
            else:
                new_win, old_win, draws = self._contest(test_number=self.contest_num)
            all_ = new_win + old_win + draws
            sum_ = new_win + old_win
            clean_rate = new_win / sum_ if sum_ != 0 else -1
            self.swanlab.log({
                "新模型获胜局数": new_win, "旧模型获胜局数": old_win, "和棋数": draws, "胜率": new_win / all_,
                "纯净胜率（-1不存在）": clean_rate
            })
            if sum_ == 0 or new_win / sum_ < 0.6:
                print(f"🐑 REJECT Win Rate {new_win / all_}, draws: {draws}")
                self.training_network.load(key="before_train.pt")
                self.swanlab.log({"is_update": 0})
            else:
                print(f"🍤 ACCEPT, Win Rate {new_win / sum_} model saved, draws: {draws}")
                self.training_network.save(epoch, key="best.pt")
                self.swanlab.log({"is_update": 1})

            self.training_network.train()
