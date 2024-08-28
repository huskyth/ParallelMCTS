import copy

import numpy as np
from chess.common import MOVE_TO_INDEX_DICT, MAX_STEPS
from mcts.node import Node
import torch

from network_wrapper import ChessNetWrapper


class MCTS:
    def __init__(self, predict, simulate_time=None):
        self.root = Node(1)
        self.model_predict = predict
        self.simulate_times = 1600 if simulate_time is None else simulate_time
        self.max_h = 0

    def _simulate(self, state, idx):
        current_node = self.root
        counter = 0
        while True:
            if current_node.is_leaf() or counter > MAX_STEPS:
                break

            action, current_node = current_node.select()
            state.do_action(action)
            counter += 1
        if self.max_h < counter:
            self.max_h = counter

        is_end, winner = state.is_end()
        if is_end is True:
            assert winner is not None
            value = 1 if winner == state.get_current_player() else -1
            current_node.update(-value)
        else:
            if counter > MAX_STEPS:
                print(f"COUNT LARGE THAN {MAX_STEPS}")
                current_node.update(0)
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

    def predict(self, state):
        return self.model_predict(state)

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
        self.max_h = 0
        for i in range(self.simulate_times):
            state_copy = copy.deepcopy(state)
            self._simulate(state_copy, i)
        probability = np.array([item.visit for item in self.root.children.values()])

        if is_greedy:
            max_visit = np.max(probability)
            probability = np.where(probability == max_visit, 1, 0)

        visit_list = probability / probability.sum()
        return visit_list


if __name__ == '__main__':
    n = ChessNetWrapper()
    m = MCTS(n.predict)
    state_test = torch.randn((8, 1, 7, 7))
    y = m.model_predict(state_test)
    for v, p in zip(*m.model_predict(state_test)):
        print(v, p)
    print(y)
    a = np.arange(8).reshape(8, 1)
    b = np.arange(8 * 72).reshape(8, 72)
    for v, p in zip(a, b):
        print(v, p)
