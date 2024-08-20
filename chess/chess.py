import numpy as np

from chess_board import ChessBoard
from common import from_array_to_input_tensor, ARRAY_TO_IMAGE
import torch


class Chess(ChessBoard):
    def __init__(self):
        super().__init__()
        self.current_player = 1

    def is_end(self):
        winner = self.check_winner()
        is_end = winner is not None
        return is_end, winner

    def get_torch_state(self):
        temp = from_array_to_input_tensor(self.pointStatus)
        state0 = (temp > 0).float()
        state1 = (temp < 0).float()
        if self.current_player == -1:
            state0, state1 = state1, state0
        return torch.cat([state0, state1], dim=1)

    def get_board(self):
        numpy_array = np.array(self.pointStatus)
        if isinstance(numpy_array, list):
            numpy_array = np.array(numpy_array)
        assert len(numpy_array) == 21
        assert isinstance(numpy_array, np.ndarray)
        input_tensor = np.zeros((7, 7))
        for i, chessman in enumerate(numpy_array):
            row, column = ARRAY_TO_IMAGE[i]
            input_tensor[row, column] = chessman
        return input_tensor

    def do_action(self, action):
        self.execute_move(action, self.current_player)
        self.current_player *= -1

    def get_current_player(self):
        return self.current_player

    def reset(self):
        self.init_point_status()
        self.current_player = 1


if __name__ == '__main__':
    c = Chess()
    c.get_torch_state()
