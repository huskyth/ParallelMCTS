from constants import ROOT_PATH
from game.chess.common import MAX_HISTORY_STEPS
from models.models import GameNet
from models.wrapper import Wrapper


class ChessNetWrapper(Wrapper):
    MODEL_SAVE_PATH = ROOT_PATH / "checkpoints" / "WMChess"
    if not MODEL_SAVE_PATH.exists():
        MODEL_SAVE_PATH.mkdir()

    def __init__(self):
        self.net = GameNet(3 + MAX_HISTORY_STEPS, 7, 72)
        super().__init__(self.net)


if __name__ == '__main__':
    import swanlab
    from pickle import Pickler, Unpickler
    with open(ChessNetWrapper.MODEL_SAVE_PATH / "train_history.examples", "rb") as f:
        temp = Unpickler(f).load()
    c = ChessNetWrapper()
    swanlab.login(api_key="rdGaOSnlBY0KBDnNdkzja")
    swan = swanlab.init(project="ChessGameSP", logdir=ROOT_PATH / "logs")
    train_sample = []
    import numpy as np
    np.random.shuffle(train_sample)
    for x in temp:
        train_sample.extend(x)
    c.train_net(train_sample, swan)
    c.save(0)
