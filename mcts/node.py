import numpy as np


class Node:
    def __init__(self, p):
        self.p = p
        self.q = 0
        self.visit = 0
        self.c = 2
        self.children = {}
        self.parent = None

    def get_value(self):
        father_visit = self.parent.visit
        value = self.q + self.c * self.p * father_visit / (1 + self.visit) ** 0.5
        if isinstance(value, np.ndarray):
            value = value.item()
        return value

    def expand(self, probability):
        for idx, v in enumerate(probability):
            temp = Node(v)
            temp.parent = self
            self.children[idx] = temp

    def select(self, mode):
        values = [item.get_value() for _, item in self.children.items()]
        childrens = [item for _, item in self.children.items()]
        best_idx = np.argmax(values)
        item = childrens[best_idx]
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
