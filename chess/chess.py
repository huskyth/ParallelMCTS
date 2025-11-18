from chess.chess_board import ChessBoard
from chess.common import from_array_to_input_tensor


class Chess(ChessBoard):
    def __init__(self, start_player=1):
        super().__init__()
        self.current_player = start_player

    def is_end(self):
        winner = self.check_winner()
        is_end = winner is not None
        return is_end, winner

    def get_torch_state(self):
        """
            得到棋盘的张量
            :return:
        """
        return from_array_to_input_tensor(self.pointStatus, self.current_player)

    def do_action(self, action):
        self.execute_move(action, self.current_player)
        self.current_player *= -1

    def get_current_player(self):
        return self.current_player

    def reset(self, start_player=1):
        self.init_point_status()
        self.current_player = start_player
