import copy

import numpy as np

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
            value = 1 if winner == state.current_play() else -1
        else:
            value, probability = self.predict(state.get_torch_state())
            current_node.expand(probability)
        current_node.update(-value)

    def update_tree(self, move):
        if move == -1:
            self.root = Node(1)
        else:
            self.root = self.root.children[move]
            self.root.parent = None

    def get_action_probability(self, state, is_greedy):
        for i in range(self.simulate_times):
            state = copy.deepcopy(state)
            self._simulate(state)

        probability = np.array([item.visit for item in self.root.children])

        if is_greedy:
            max_visit = np.max(probability)
            probability = np.where(probability == max_visit, 1, 0)
            return probability

        visit_list = probability / probability.sum()
        return visit_list


if __name__ == '__main__':
    m = MCTS(None)
    y = m.get_action_probability(None, False)
    print(y)
