import random
from copy import deepcopy
import numpy as np
import torch

from game.abstract_state import AbstractState

X, O, EMPTY = 'X', 'O', None
BOARD_SIZE = 3
player2sign = {1: X, -1: O}
sign2value = {X: 1, O: -1, EMPTY: 0}


# Tic Tac Toe board class
class Board:  # must contain (win,draw,player,board,valid actions,move) for mcts
    # create constructor (init board class instance)
    def __init__(self, board=None):
        # define players
        self.player = 1  # 1 for first player; -1 for second player
        if not board:
            board = [[EMPTY for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]
        self.board = deepcopy(board)

    # make move
    def move(self, action):
        row, col = action
        # create new board instance that inherits from the current state
        next_state = Board(self.board)

        assert next_state.board[row][col] is EMPTY
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
    @property
    def move_to_index(self) -> dict:
        return self._move_to_index

    @property
    def index_to_move(self) -> dict:
        return self._index_to_move

    def reset(self, start_player=1) -> None:
        self.state = Board()
        self.board = self.state.board
        self.winner = self.check_winner()
        self.state.player = start_player
        self.player = self.state.player
        self.left = 9
        self.last_action = None

    def get_current_player(self) -> int:
        return self.player

    def do_action(self, action: int) -> None:
        row, col = action // 3, action % 3
        self.move((row, col))

    def is_end(self) -> (bool, int):
        return self.winner is not None, self.winner

    def get_torch_state(self) -> torch.Tensor:
        board = deepcopy(self.board)
        for i in range(BOARD_SIZE):
            for j in range(BOARD_SIZE):
                board[i][j] = sign2value[board[i][j]]

        player_state = torch.tensor(board, dtype=torch.float32)[:, :, None]
        other_state = torch.tensor(board, dtype=torch.float32)[:, :, None]

        player_state[player_state == -self.player] = 0
        player_state[player_state == self.player] = 1

        other_state[other_state == self.player] = 0
        other_state[other_state == -self.player] = 1

        player = torch.ones(BOARD_SIZE, BOARD_SIZE, 1) * (1 if self.player == 1 else 0)

        last_action = torch.zeros(player.shape)
        if self.last_action is not None:
            last_action[self.last_action] = 1

        state = torch.cat([player_state, other_state, last_action], dim=2).float()

        return state.cuda() if torch.cuda.is_available() else state

    def get_legal_moves(self, player) -> list:
        return self.available_actions()

    def __init__(self, board=None, is_render=False):
        self.state = board if board else Board()
        self.board = self.state.board
        self.winner = self.check_winner()
        self.player = self.state.player
        self.left = 9
        self._index_to_move = {
            i * 3 + j: (i, j) for i in range(BOARD_SIZE) for j in range(BOARD_SIZE)
        }
        self._move_to_index = {
            (i, j): i * 3 + j for i in range(BOARD_SIZE) for j in range(BOARD_SIZE)
        }
        self.last_action = None
        self.is_render = is_render

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

    def render(self, title):
        if not self.is_render:
            return
        print('\n' + '*' * 20 + title + '*' * 20)
        print("board:")
        print("   0 1 2")
        for i in range(BOARD_SIZE):
            print(i, end="  ")
            for j in self.board[i]:
                print(j if j is not None else '-', end=" ")
            print()
        print('*' * 20 + title + '*' * 20)

    def move(self, action):
        self.left -= 1
        self.state = self.state.move(action)
        self.last_action = action
        self.board = self.state.board
        self.winner = self.check_winner()
        self.player = -self.player
        if self.winner is None and self.left == 0:
            self.winner = 0

    def move_random(self):
        tuple_act = random.choice(self.get_legal_moves(self.get_current_player()))
        max_act = tuple_act[0] * 3 + tuple_act[1]
        return max_act

    def top_buttom(self, state, probability):
        state[:, :, 0] = torch.flip(state[:, :, 0], (0,))
        state[:, :, 1] = torch.flip(state[:, :, 1], (0,))
        probability = np.reshape(np.flip(probability.reshape(3, 3), (0,)), (9,))
        return state, probability

    def left_right(self, state, probability):
        state[:, :, 0] = torch.flip(state[:, :, 0], (1,))
        state[:, :, 1] = torch.flip(state[:, :, 1], (1,))
        probability = np.reshape(np.flip(probability.reshape(3, 3), (1,)), (9,))
        return state, probability

    def center(self, state, probability):
        s, p = self.top_buttom(state, probability)
        return self.left_right(s, p)


if __name__ == '__main__':
    game = TicTacToe()
    game.do_action(5)
    game.do_action(6)
    game.do_action(1)
    temp = game.get_torch_state()
    print(temp[:, :, 0])
    print(temp[:, :, 1])
    print(temp[:, :, 2])
    print(temp[:, :, 3])
