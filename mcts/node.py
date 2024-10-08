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
        return self.q + self.c * self.p * father_visit / (1 + self.visit) ** 0.5

    def expand(self, probability):
        for idx, v in enumerate(probability):
            temp = Node(v)
            temp.parent = self
            self.children[idx] = temp

    def select(self):
        best_idx, best_u = 0, -float("inf")
        for key, item in self.children.items():
            if item.get_value() > best_u:
                best_idx = key
                best_u = item.get_value()
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
