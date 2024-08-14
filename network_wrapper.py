from collections import deque

import numpy as np
import torch

from chess.common import MODEL_SAVE_PATH
from models import ChessNet
import torch.nn.functional as F
from torch.optim import Adam


class ChessNetWrapper:
    def __init__(self):
        self.net = ChessNet()
        self.is_cuda = torch.cuda.is_available()
        self.opt = Adam(self.net.parameters(), lr=1e-3, weight_decay=1e-2)
        self.epoch = 10
        self.batch = 1

    @torch.no_grad()
    def predict(self, state):
        if len(state.shape) == 2:
            state = state.unsqueeze(0).unsqueeze(0)
        v, p = self.net(state)
        return v.detach().cpu().numpy(), p.detach().cpu().numpy()

    def smooth_l1(self, input_tensor, target_tensor):
        return F.smooth_l1_loss(input_tensor, target_tensor)

    def cross_entropy(self, p, p_target):
        return F.cross_entropy(p, p_target)

    def train(self, train_sample):
        n = len(train_sample)
        state, probability, _, value = list(zip(*train_sample))
        state = torch.tensor(state).float()
        state = state.cuda() if self.is_cuda else state
        state = state.unsqueeze(1)
        probability = torch.tensor(probability).float()
        probability = probability.cuda() if self.is_cuda else probability
        value = torch.tensor(value).float()
        value = value.cuda() if self.is_cuda else value

        batch_number = n // self.batch
        for epoch in range(self.epoch):
            for step in range(batch_number):
                start = step * self.batch
                state = state[start:start + self.batch, :, :, :]
                probability = probability[start:start + self.batch, :]
                value = value[start:start + self.batch]

                v_predict, p_predict = self.net(state)
                value_loss = self.smooth_l1(v_predict, value)
                probability_loss = self.cross_entropy(probability, p_predict)
                loss = value_loss + probability_loss
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()

    def save(self, key):
        torch.save({"state_dict": self.net.state_dict()}, str(MODEL_SAVE_PATH / key))

    def load(self, key):
        model = torch.load(str(MODEL_SAVE_PATH / key))
        self.net.load_state_dict(model["state_dict"])


if __name__ == '__main__':
    cn = ChessNetWrapper()
    a = [np.random.random((7, 7)), np.random.random(72), 1, 1]
    b = [np.random.random((7, 7)), np.random.random(72), -1, -1]
    c = [np.random.random((7, 7)), np.random.random(72), 1, -1]
    tp = deque(maxlen=10)
    tp.append(a)
    tp.append(b)
    tp.append(c)
    np.random.shuffle(tp)
    cn.train(tp)
    state_, probability_, player, value_ = list(zip(*tp))
    print(len(state_))
    print(len(probability_))
    print(player)
    print(value_)
