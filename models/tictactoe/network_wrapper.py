from constants import ROOT_PATH
from models.models import GameNet
from models.wrapper import Wrapper


class TictactoeNetWrapper(Wrapper):
    MODEL_SAVE_PATH = ROOT_PATH / "tictactoe" / "checkpoints"
    if not MODEL_SAVE_PATH.exists():
        MODEL_SAVE_PATH.mkdir()

    def __init__(self):
        self.net = GameNet(2, 3, 9)
        super().__init__(self.net)
