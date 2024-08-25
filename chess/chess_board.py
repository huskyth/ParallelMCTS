import copy

from common import GAME_MAP, \
    LENGTH_OF_BOARD, BLACK, WHITE, DISTANCE, get_neighbours, shiftOutChessman, \
    INDEX_TO_MOVE_DICT


class ChessBoard:

    def __init__(self):
        self.gameMap = []
        self.pointStatus = []
        self.distance = []
        self.status = None
        self.whiteNum = 6
        self.blackNum = 6
        self.init_distance()
        self.init_point_status()
        self.init_game_map()
        self.is_simple = True
        self.last_action = (-1, -1)

    def get_game_map(self):
        return self.gameMap

    def get_point_status(self):
        return self.pointStatus

    def get_distance(self):
        return self.distance

    def init_distance(self):
        self.distance = DISTANCE

    def init_point_status(self):
        self.pointStatus = []
        black = [0, 1, 2, 3, 4, 8]
        white = [7, 11, 12, 13, 14, 15]
        for x in range(LENGTH_OF_BOARD):
            self.pointStatus.append(0)
        for x in black:
            self.pointStatus[x] = BLACK
        for x in white:
            self.pointStatus[x] = WHITE

    def init_chessman_num(self):
        self.whiteNum = 6
        self.blackNum = 6

    def init_game_map(self):
        self.gameMap = GAME_MAP

    def get_legal_moves(self, player):
        assert player in [WHITE, BLACK]
        legal_moves_list = []
        for from_point_idx, chessman in enumerate(self.pointStatus):
            if chessman != player:
                continue
            to_point_idx_list = get_neighbours(from_point_idx, self.distance)
            for to_point_idx in to_point_idx_list:
                to_point = self.pointStatus[to_point_idx]
                if to_point != 0:
                    continue
                legal_moves_list.append((from_point_idx, to_point_idx))
        return legal_moves_list

    def execute_move(self, move, color):
        if isinstance(move, int):
            move = INDEX_TO_MOVE_DICT[move]
        from_int, to_int = move
        assert color == WHITE or color == BLACK
        assert self.pointStatus[from_int] == color
        assert self.pointStatus[to_int] == 0
        assert self.distance[from_int][to_int] == 1
        self.pointStatus[from_int] = 0
        self.pointStatus[to_int] = color
        self.last_action = (from_int, to_int)
        bake_point_status = copy.deepcopy(self.pointStatus)
        self.pointStatus = shiftOutChessman(
            bake_point_status, self.distance)

    def check_winner(self):
        black_num = 0
        white_num = 0
        winner = None
        for color in self.pointStatus:
            if color == BLACK:
                black_num += 1
            elif color == WHITE:
                white_num += 1
        if self.is_simple:
            if white_num > black_num:
                return WHITE
            elif white_num < black_num:
                return BLACK
            return None
        if black_num < 3 or white_num < 3:
            if black_num < 3:
                winner = WHITE
            else:
                winner = BLACK

        return winner
