from models import ChessNet


class ChessNetWrapper:
    def __init__(self):
        self.net = ChessNet()

    def predict(self, state):
        pass

    def train(self, train_sample):
        pass

    def save(self, key):
        pass

    def load(self, key):
        pass
