import time

import numpy as np


class Node:
    def __init__(self, p):
        self.p = p
        self.q = 0
        self.visit = 0
        self.c = 5
        self.children = {}
        self.parent = None

    def get_value(self):
        father_visit = self.parent.visit
        return self.q + self.c * self.p * (father_visit ** 0.5) / (
                1 + self.visit)

    def expand(self, probability):

        dirichlet_noise = 0.25 * np.random.dirichlet(0.3 * np.ones(len(np.nonzero(probability)[0])))
        probability *= 0.75
        j = 0
        for idx, v in enumerate(probability):
            if v > 0:
                v = dirichlet_noise[j] + v
                j += 1
            temp = Node(v)
            temp.parent = self
            self.children[idx] = temp

    def select(self):
        best_idx, best_u = None, -float("inf")
        all_zero = True
        for key, item in self.children.items():
            if item.p == 0:
                continue
            all_zero = False
            if item.get_value() > best_u:
                best_idx = key
                best_u = item.get_value()
        if best_idx is None:
            with open(f"log_error_{time.time()}.txt", "a") as f:
                f.write(str([x.p for x in self.children.values()]))
        assert best_idx is not None, f"all_zero {all_zero}"
        return best_idx, self.children[best_idx]

    def _update(self, value):
        self.visit += 1
        self.q += (value - self.q) / self.visit

    def update(self, value):
        if self.parent is not None:
            self.parent.update(-value)
        self._update(value)

    def is_leaf(self):
        return self.children == {}


if __name__ == '__main__':
    pass
