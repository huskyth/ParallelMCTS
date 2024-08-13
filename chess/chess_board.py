from chess.common import GAME_MAP, LENGTH_OF_BOARD, BLACK, WHITE, DISTANCE


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
