import copy

import numpy as np

from chess.common import MOVE_TO_INDEX_DICT
from mcts.node import Node


class MCTS:
    def __init__(self, predict):
        self.root = Node(1)
        self.predict = predict
        self.simulate_times = 1600

    def _simulate(self, state):
        current_node = self.root
        while True:
            if current_node.is_leaf():
                break

            action, current_node = current_node.select()
            state.do_action(action)

        is_end, winner = state.is_end()
        if is_end is True:
            assert winner is not None
            value = 1 if winner == state.get_current_player() else -1
        else:
            value, probability = self.predict(state.get_torch_state())
            probability = probability[0]
            available_action = state.get_legal_moves(state.get_current_player())
            available_ = set()
            for move in available_action:
                available_.add(MOVE_TO_INDEX_DICT[move])
            for idx, p in enumerate(probability):
                if idx not in available_:
                    probability[idx] = 0
            probability /= probability.sum()
            # TODO: test it
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
                self.root = Node(1)

    def get_action_probability(self, state, is_greedy):

        for i in range(self.simulate_times):
            state_copy = copy.deepcopy(state)
            self._simulate(state_copy)

        probability = np.array([item.visit for item in self.root.children.values()])

        if is_greedy:
            max_visit = np.max(probability)
            probability = np.where(probability == max_visit, 1, 0)
            return probability

        visit_list = probability / probability.sum()
        return visit_list
