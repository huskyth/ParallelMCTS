from copy import deepcopy

import torch

from game.abstract_state import AbstractState

X, O, EMPTY = 'X', 'O', None
BOARD_SIZE = 3
player2sign = {1: X, -1: O}


# Tic Tac Toe board class
class Board:  # must contain (win,draw,player,board,valid actions,move) for mcts
    # create constructor (init board class instance)
    def __init__(self, board=[[EMPTY for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]):
        # define players
        self.player = 1  # 1 for first player; -1 for second player
        self.board = deepcopy(board)

    # make move
    def move(self, action):
        row, col = action
        # create new board instance that inherits from the current state
        next_state = Board(self.board)

        # make move
        next_state.board[row][col] = player2sign[self.player]

        # swap players
        next_state.player = -self.player

        # return new board state
        return next_state

    # get whether the game is drawn
    def is_draw(self):
        # loop over board squares
        for row in range(BOARD_SIZE):
            for col in range(BOARD_SIZE):
                # empty square is available
                if self.board[row][col] == EMPTY:
                    # this is not a draw
                    return False

        # by default we return a draw
        return True

    # get whether the game is won
    def is_win(self):
        # check row
        for i in range(3):
            base_sign = self.board[i][1]
            if base_sign != EMPTY and self.board[i][0] == base_sign and base_sign == self.board[i][2]:
                return True
        # check column
        for i in range(3):
            base_sign = self.board[1][i]
            if base_sign != EMPTY and self.board[0][i] == base_sign and base_sign == self.board[2][i]:
                return True
        # check diagnals
        base_sign = self.board[1][1]
        if base_sign != EMPTY:
            if self.board[0][0] == base_sign and self.board[2][2] == base_sign:
                return True
            elif self.board[0][2] == base_sign and self.board[2][0] == base_sign:
                return True
        return False

    # generate legal moves to play in the current position
    def generate_actions(self):
        # define states list (move list - list of available actions to consider)
        actions = []

        # loop over board rows
        for row in range(BOARD_SIZE):
            # loop over board columns
            for col in range(BOARD_SIZE):
                # make sure that current square is empty
                if self.board[row][col] == EMPTY:
                    # append available row, col to action list
                    actions.append((row, col))

        # return the list of available actions (tuple)
        return actions


class TicTacToe(AbstractState):
    def get_current_player(self) -> int:
        pass

    def do_action(self, action: int) -> None:
        pass

    def is_end(self) -> (bool, int):
        pass

    def get_torch_state(self) -> torch.Tensor:
        pass

    def get_legal_moves(self) -> list:
        pass

    def __init__(self, board=Board()):
        self.state = board
        self.board = self.state.board
        self.winner = self.check_winner()
        self.player = self.state.player
        self.left = 9

    def available_actions(self):
        return self.state.generate_actions()

    def check_winner(self):
        for i in range(3):
            base_sign = self.board[i][1]
            if base_sign != EMPTY and self.board[i][0] == base_sign and base_sign == self.board[i][2]:
                if base_sign == X:
                    return 1
                else:
                    return -1
        for i in range(3):
            base_sign = self.board[1][i]
            if base_sign != EMPTY and self.board[0][i] == base_sign and base_sign == self.board[2][i]:
                if base_sign == X:
                    return 1
                else:
                    return -1
        base_sign = self.board[1][1]
        if base_sign != EMPTY:
            if self.board[0][0] == base_sign and self.board[2][2] == base_sign:
                if base_sign == X:
                    return 1
                else:
                    return -1
            elif self.board[0][2] == base_sign and self.board[2][0] == base_sign:
                if base_sign == X:
                    return 1
                else:
                    return -1
        return None

    def render(self):
        print()
        print("board:")
        print("   0 1 2")
        for i in range(BOARD_SIZE):
            print(i, end="  ")
            for j in self.board[i]:
                print(j if j is not None else '-', end=" ")
            print()
        print()

    def move(self, action):
        self.left -= 1
        self.state = self.state.move(action)
        self.board = self.state.board
        self.winner = self.check_winner()
        self.player = -self.player
        if self.winner is None and self.left == 0:
            self.winner = 0
