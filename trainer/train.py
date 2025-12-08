import os
from collections import deque
import swanlab
import numpy as np
import torch
from constants import ROOT_PATH
from game.chess.chess import Chess
from mcts.pure_mcts import MCTS
from models.wm_model.network_wrapper import ChessNetWrapper

from utils.concurrent_tool import ConcurrentProcess


class Trainer:
    def __init__(self, train_config=None, use_swanlab=True, mode='train', number_of_self_play=5, number_of_contest=5,
                 use_pool=False, is_render=False, is_data_augment=False, is_image_show=False):
        if use_swanlab:
            swanlab.login(api_key="rdGaOSnlBY0KBDnNdkzja")
            self.swanlab = swanlab.init(project="Chess", logdir=ROOT_PATH / "logs")
        else:
            self.swanlab = None
        self.train_config = train_config

        self.is_data_augment = is_data_augment
        self.is_image_show = is_image_show

        self.train_sample = deque(maxlen=5000)
        self.is_render = is_render
        self.use_pool = use_pool
        self.current_play_turn = 0
        if use_pool:
            self.self_play_parallel_num = number_of_self_play
            self.self_play_processor = ConcurrentProcess(self.self_play_parallel_num)
            self.contest_parallel_num = number_of_contest
            self.contest_processor = ConcurrentProcess(self.contest_parallel_num)
        else:
            self.self_play_num = number_of_self_play
            self.contest_num = number_of_contest

        self.training_network = ChessNetWrapper()

    def _collect_concurrent(self):
        mcts = self.abstract_game.mcts
        state = self.abstract_game.state
        param = (self.current_play_turn, mcts, state, False, self.is_data_augment, self.is_image_show)

        result = self.self_play_processor.process(self._self_play, *param)
        self.current_play_turn += self.self_play_parallel_num
        ret = []
        for r in result:
            ret.extend(r)
        return ret

    def _collect(self):
        sample = []
        mcts1 = MCTS(self.training_network.predict, mode='train', name="自我对弈玩家1")
        mcts2 = MCTS(self.training_network.predict, mode='train', name="自我对弈玩家2")
        state = Chess(is_render=self.is_render)
        for i in range(self.self_play_num):
            temp = self._self_play(self.current_play_turn, mcts1, mcts2, state, self.is_render, self.is_data_augment,
                                   self.is_image_show)
            self.current_play_turn += 1
            sample.extend(temp)
        return sample

    @staticmethod
    def _self_play(current_play_turn, mcts1, mcts2, state, is_render, is_data_augment, is_image_show):
        train_sample = []
        turn = 0
        mcts1.update_tree(-1)
        mcts2.update_tree(-1)
        f_p = 1 if (current_play_turn + 1) % 2 == 0 else -1
        player_list = [mcts1, None, mcts2]
        state.reset(f_p)
        start_player = f_p
        print(
            f"😊 开始第{current_play_turn + 1}轮self_play，先手是 {f_p}，"
            f"name是 {player_list[start_player + 1].name}，"
            f"进程ID {os.getpid()}")

        state.image_show(f"测试局面", is_image_show)
        while not state.is_end()[0]:
            turn += 1
            if turn % 100 == 0:
                print(f"😊 第{current_play_turn + 1}次self_play 共进行 {turn} 轮")
            probability = player_list[start_player + 1].get_action_probability(state=state, is_greedy=False)
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

    def _contest_concurrent(self, mode):
        new_win, old_win, draws = 0, 0, 0
        new_player = self.abstract_game.mcts
        old_mcts = self.abstract_game.random_mcts
        if mode == 'train':
            self.abstract_game.random_network.load("before_train.pt")
        state = self.abstract_game.state
        param = (0, state, new_player, old_mcts, None)
        ret = self.contest_processor.process(self._contest_one_time, *param)
        for item in ret:
            new_win_, old_win_, draws_, length_of_turn_ = item
            new_win += new_win_
            old_win += old_win_
            draws += draws_
            print(f"♬ 本局进行了{length_of_turn_}轮")

        return new_win, old_win, draws

    def _contest(self, test_number=1000):
        new_player = MCTS(self.training_network.predict, mode='test', name="当前训练玩家")
        state = Chess(is_render=self.is_render)

        contest_network = ChessNetWrapper()
        try:
            contest_network.load("best.pt")
        except Exception as e:
            print("best.pt模型不存在，加载before_train.pt")
            contest_network.load("latest.pt")
        last_mcts = MCTS(contest_network.predict, mode='test', name="之前最优玩家")

        wins = 0
        olds = 0
        draws = 0
        for i in range(test_number):
            new_win, old_win, draw, length_of_turn = self._contest_one_time(i, state, new_player, last_mcts,
                                                                            self.is_image_show)
            print(f"♬ 本局进行了{length_of_turn}轮\n")
            wins += new_win
            olds += old_win
            draws += draw
        return wins, olds, draws

    @staticmethod
    def _contest_one_time(i, state, new_player, last_mcts, is_image_show):

        new_win, old_win, draws = 0, 0, 0
        new_player.update_tree(-1)
        last_mcts.update_tree(-1)
        player_list = [last_mcts, None, new_player]
        current_player = 1 if i % 2 == 0 else -1
        start_player = current_player
        state.reset(start_player)
        print(f"\n🌟 start {i}th contest, first hand is {start_player}, {player_list[start_player + 1].name}")
        length_of_turn = 0
        max_turn = 1000
        state.render("初始化局面")
        state.image_show("Contest", is_image_show)
        while not state.is_end()[0]:
            length_of_turn += 1
            if length_of_turn >= max_turn:
                print(f"🍑 draws is 1, old win is 0, new win is 0")
                return 0, 0, 1, length_of_turn
            player = player_list[current_player + 1]
            if player is None:
                max_act = state.move_random()
            else:
                probability_new = player.get_action_probability(state, False)
                max_act = np.argmax(probability_new).item()
                max_act = np.random.choice(len(probability_new), p=probability_new)
            p_name = player.name if player else '随机玩家'
            state.render(
                f"Step {length_of_turn} - 当前玩家 {p_name} {state.get_current_player()}, 执行 {state.index_to_move[max_act]}")
            state.do_action(max_act)
            state.render(f"Step {length_of_turn} - 当前玩家 {p_name} 索引{-state.get_current_player()}执行后的局面")
            new_player.update_tree(max_act)
            last_mcts.update_tree(max_act)
            current_player *= -1
            state.image_show("Contest", is_image_show)

        _, winner = state.is_end()
        if winner == 1:
            new_win += 1
        elif winner == -1:
            old_win += 1
        elif winner == 0:
            draws += 1

        print(f"🍑 draws is {draws}, old win is {old_win}, new win is {new_win}")
        return new_win, old_win, draws, length_of_turn

    def test(self, test_number):
        self.training_network.load("best.pt")
        new_player = MCTS(self.training_network.predict, mode='test', name="最佳玩家")
        state = Chess(is_render=self.is_render)

        contest_network = ChessNetWrapper()
        random_mcts = MCTS(contest_network.predict, mode='test', name="随机玩家")

        wins = 0
        olds = 0
        draws = 0
        for i in range(test_number):
            new_win, old_win, draw, length_of_turn = self._contest_one_time(i, state, new_player, random_mcts,
                                                                            self.is_image_show)
            print(f"♬ 本局进行了{length_of_turn}轮\n")
            wins += new_win
            olds += old_win
            draws += draw
        all_ = wins + olds + draws
        self.swanlab.log({
            "win_new": wins, "win_random": olds, "draws": draws, "win_rate": wins / all_,
            "pure_rate": wins / (wins + olds)
        })
        print(f"🍤 Win Rate {wins / all_}")

    def play(self, current_player="AI"):
        if self.abstract_game.game == 'WMChess':
            self._wm_play()
            return
        if current_player not in ["AI", "Human"]:
            raise ValueError("current_player must be 'AI' or 'Human'")
        self.abstract_game.network.load("best.pt")
        ai = self.abstract_game.mcts
        ai.mode = 'test'
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
        ai.mode = 'train'
        if winner == 0:
            print("和棋")
        elif winner == 1:
            print(f"{start_player} 赢了")
        elif winner == -1:
            print(f"{ano_player} 赢了")

    def _wm_play(self):
        from game.chess.wm_chess_gui import WMChessGUI
        state = self.abstract_game.state
        self.abstract_game.random_network.load("best.pt")
        self.abstract_game.random_network.eval()
        mcts = self.abstract_game.random_mcts
        state.reset()
        wm = WMChessGUI(mcts, state)
        wm.start()

    def learn(self):

        start_epoch = self.training_network.try_load()

        is_trained = False
        for epoch in range(start_epoch, self.train_config.epoch):

            if self.use_pool:
                train_sample = self._collect_concurrent()
            else:
                train_sample = self._collect()

            self.train_sample.extend(train_sample)

            if len(self.train_sample) >= 1000:
                is_trained = True
                print(f"start training... size of train_sample: {len(self.train_sample)}")
                np.random.shuffle(self.train_sample)
                self.training_network.save(epoch, key="before_train.pt")
                stat = self.training_network.train_net(self.train_sample)
                self.training_network.save(epoch)
                for sta in stat:
                    self.swanlab.log(sta)

            if (epoch + 1) % self.train_config.test_rate == 0 and is_trained:
                self.training_network.eval()
                if self.use_pool:
                    new_win, old_win, draws = self._contest_concurrent(mode='train')
                else:
                    new_win, old_win, draws = self._contest(test_number=self.contest_num)
                all_ = new_win + old_win + draws
                self.swanlab.log({
                    "win_new": new_win, "win_random": old_win, "draws": draws, "win_rate": new_win / all_
                })
                if new_win > old_win:
                    print(f"🍤 ACCEPT, Win Rate {new_win / all_} model saved")
                    self.training_network.save(epoch, key="best.pt")
                else:
                    print(f"🐑 REJECT Win Rate {new_win / all_}")
                    self.training_network.load(key="before_train.pt")
                self.training_network.train()
