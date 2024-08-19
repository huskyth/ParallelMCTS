import time


class Node:
    def __init__(self, p):
        self.p = p
        self.q = 0
        self.visit = 0
        self.c = 5
        self.children = {}
        self.parent = None
        self.visual_loss = 0

    def get_value(self, visual_loss_c):
        father_visit = self.parent.visit
        return self.q + self.c * self.p * father_visit / (
                1 + self.visit) ** 0.5 - visual_loss_c * self.visual_loss / (
                1 + self.visit)

    def expand(self, probability):
        for idx, v in enumerate(probability):
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
            if item.get_value(self.visual_loss) > best_u:
                best_idx = key
                best_u = item.get_value(self.visual_loss)
        if best_idx is None:
            with open(f"log_error_{time.time()}.txt", "a") as f:
                f.write(str([x.p for x in self.children.items()]))
        assert best_idx is not None, f"all_zero {all_zero}"
        self.children[best_idx].visual_loss += 1
        return best_idx, self.children[best_idx]

    def _update(self, value):
        self.visual_loss -= 1
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
