import numpy as np


class Node:
    def __init__(self, p):
        self.p = p
        self.q = 0
        self.visit = 0
        self.c = 1
        self.children = {}
        self.parent = None

    def get_value(self, state, idx):
        father_visit = self.parent.visit
        value = self.q + self.c * self.p * father_visit ** 0.5 / (1 + self.visit)
        if isinstance(value, np.ndarray):
            value = value.item()

        state.render(
            f"当前 q = {self.q}, c = {self.c}, p = {self.p}, f_visit = {father_visit}, visit = {self.visit},"
            f" 当前玩家 {state.get_current_player()} value = {value}, 这个move = {idx}")
        return value

    def expand(self, probability):
        for idx, v in enumerate(probability):
            temp = Node(v)
            temp.parent = self
            self.children[idx] = temp

    def select(self, mode, state):
        values = [item.get_value(state, idx) for idx, item in self.children.items() if item.p > 0]
        items = [item for _, item in self.children.items() if item.p > 0]
        its = [idx for idx, item in self.children.items() if item.p > 0]
        max_idx = np.argmax(values)
        item = items[max_idx]
        best_idx = its[max_idx]

        return best_idx, item

    def _update(self, value):
        self.visit += 1
        self.q += (value - self.q) / self.visit

    def update(self, value):
        if self.parent is not None:
            self.parent.update(-value)
        self._update(value)

    def is_leaf(self):
        return self.children == {}
