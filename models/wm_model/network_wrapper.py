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
