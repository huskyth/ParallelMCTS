import numpy as np

from chess.chess import Chess
from constants import ACTION_SIZE
from mcts.pure_mcts import MCTS
from network import ChessNet


class Trainer:
    def __init__(self):
        self.epoch = 100
        self.test_rate = 5
        self.greedy_times = 5
        self.dirichlet_rate = 1 - 0.25
        self.dirichlet_probability = 1 - 0.25
        self.network = ChessNet()
        self.mcts = MCTS(self.network.predict)
        self.state = Chess()

    def _collect(self):
        return None

    def _play(self):
        step = 0
        train_sample = []
        self.mcts.update_tree(-1)
        self.state.reset()
        while not self.state.is_end()[0]:
            step += 1
            is_greedy = step < self.greedy_times
            probability = self.mcts.get_action_probability(state=self.state, is_greedy=is_greedy)
            action = np.random.choice(probability)
            train_sample.append([self.state.get_torch_state(), probability, self.state.get_current_player()])
            self.state.do_action(action)
        _, winner = self.state.is_end()
        assert winner is not None
        for item in train_sample:
            if item[-1] == winner:
                item.append(1)
            else:
                item.append(-1)
        return train_sample

    def _contest(self):
        pass

    def learn(self):
        for epoch in range(self.epoch):
            train_sample = self._collect()

            self.network.train(train_sample)

            if (epoch + 1) % self.test_rate == 0:
                self._contest()
