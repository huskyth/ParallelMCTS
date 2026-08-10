import copy

import numpy as np

from game.chess.common import MOVE_TO_INDEX_DICT
from mcts.node import Node


class MCTS:
    def __init__(self, predict, mode='train', swanlab=None, name=None, simulate_times=1050):
        if mode not in ["train", 'test']:
            raise ValueError("mode must be 'train' or 'test'")
        self.root = Node(1)
        self.predict = predict
        self.simulate_times = simulate_times
        self.swanlab = swanlab
        self.name = name
        self.useful_sim = 0

    def _simulate(self, state):
        current_node = self.root
        while True:
            if current_node.is_leaf():
                break
            action, current_node = current_node.select()
            state.do_action(action)

        is_end, winner = state.is_end()
        if is_end is True:
            self.useful_sim += 1
            assert winner is not None
            value = 1 if winner == state.get_current_player() else -1
            if winner == 0:
                value = 0
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

    def update_tree(self):
        self.root = Node(1)


    def get_action_probability(self, state, is_greedy):
        self.useful_sim = 0
        for i in range(self.simulate_times):
            state_copy = copy.deepcopy(state)
            self._simulate(state_copy)

        probability = np.array([item.visit for item in self.root.children.values()])

        visit_list = probability / probability.sum()

        if is_greedy:
            bestAs = np.array(np.argwhere(probability == np.max(probability))).flatten()
            bestA = np.random.choice(bestAs)
            probs = [0] * len(probability)
            probs[bestA] = 1
            return np.array(probs)
        return visit_list
