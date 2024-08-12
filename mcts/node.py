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
        return max([item for idx, item in self.children.items()], key=lambda x: x.get_value())

    def _update(self, value):
        self.visit += 1
        self.q += (value - self.q) / self.visit

    def update(self, value):
        if self.parent is not None:
            self.parent.update(-value)
        self._update(value)


if __name__ == '__main__':
    pass
