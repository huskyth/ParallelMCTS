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
        childrens = [item for _, item in self.children.items() if item.p > 0]
        move_idx = [idx for idx, item in self.children.items() if item.p > 0]
        values = [item.get_value() for item in childrens]
        value_sum = sum([np.e ** value for value in values])
        probability = [np.e ** value / value_sum for value in values]
        if len(childrens) == 0:
            print(f"✨ select 中出现了问题，子节点的概率如下：\n\n {[self.children[key].p for key in self.children]} \n\n")

        if mode == "train":
            from utils.math_tool import dirichlet_noise
            probability = dirichlet_noise(probability, alpha=0.03, epison=0.3)

        import torch
        if not torch.isclose(torch.tensor(sum(probability)).float(), torch.tensor(1.0).float()):
            raise ValueError(f"probability must sum to 1, {probability} sum to {sum(probability)}")

        # best_idx = move_idx[np.argmax(probability)]
        best_idx = move_idx[np.random.choice(len(probability), 1, p=probability)[0]]
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
