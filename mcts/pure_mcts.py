import copy

import numpy as np

from mcts.node import Node


class MCTS:
    def __init__(self, predict, mode='train', swanlab=None):
        if mode not in ["train", 'test']:
            raise ValueError("mode must be 'train' or 'test'")
        self.root = Node(1)
        self.predict = predict
        self.simulate_times = 400 if mode == 'train' else 400
        self.mode = mode
        self.swanlab = swanlab
        self.max_depth = -1
        self.simulate_success_rate = 0
        self.win_rate = 0

    def _simulate(self, state):
        current_pr = state.get_current_player()
        max_depth = 0
        current_node = self.root
        while True:
            if current_node.is_leaf():
                break
            max_depth += 1
            action, current_node = current_node.select(self.mode)
            state.do_action(action)

        if max_depth > self.max_depth:
            self.max_depth = max_depth

        is_end, winner = state.is_end()
        if is_end is True:
            self.simulate_success_rate += 1
            assert winner is not None
            value = 1 if winner == state.get_current_player() else -1
            if winner == 0:
                value = 0
            if winner == current_pr:
                self.win_rate += 1
        else:
            value, probability = self.predict(state.get_torch_state())
            available_action = state.get_legal_moves(state.get_current_player())
            available_ = set()
            for move in available_action:
                available_.add(state.move_to_index[move])

            for idx, p in enumerate(probability):
                if idx not in available_:
                    probability[idx] = 0

            probability /= probability.sum()

            if probability.sum() == 0:
                print(f"✨ _simulate 中出现了问题，子节点的概率如下：\n\n {probability} \n\n")
            current_node.expand(probability)
        current_node.update(-value)

    def update_tree(self, move):
        if move not in self.root.children:
            self.root = Node(1)
        else:
            if self.root.children[move].p > 0:
                self.root = self.root.children[move]
                self.root.parent = None
            else:
                print("p is 0")
                self.root = Node(1)

    def get_action_probability(self, state, is_greedy):

        for i in range(self.simulate_times):
            state_copy = copy.deepcopy(state)
            self._simulate(state_copy)

        probability = np.array([item.visit for item in self.root.children.values()])

        explore_rate = 0
        ava_moves = state.get_legal_moves(state.get_current_player())
        ava_num = len(ava_moves)
        for idx, key in enumerate(self.root.children):
            move = state.index_to_move[idx]
            item = self.root.children[key]
            if item.visit > 0 and move in ava_moves:
                explore_rate += 1

        if self.swanlab:
            self.swanlab.log({
                "max_depth": self.max_depth, "simulate_has_result_times": self.simulate_success_rate,
                "explore_rate": explore_rate / ava_num,
                "simulate_useful_thought_rate": self.win_rate / self.simulate_success_rate if self.simulate_success_rate != 0 else -1
            })
        self.max_depth = -1
        self.simulate_success_rate = 0
        self.win_rate = 0

        if is_greedy:
            max_visit = np.max(probability)
            probability = np.where(probability == max_visit, 1, 0)
            return probability
        visit_list = probability / probability.sum()
        return visit_list
