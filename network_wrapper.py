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
        self.net = self.net.cuda() if self.is_cuda else self.net
        self.opt = Adam(self.net.parameters(), lr=1e-3, weight_decay=1e-2)
        self.epoch = 10
        self.batch = 8

    @torch.no_grad()
    def predict(self, state):
        self.net.eval()
        if len(state.shape) == 2:
            state = state.unsqueeze(0).unsqueeze(0)
        state = state.cuda() if self.is_cuda else state
        v, p = self.net(state)
        return v.detach().cpu().numpy(), p.detach().cpu().numpy()

    def smooth_l1(self, input_tensor, target_tensor):
        return F.smooth_l1_loss(input_tensor, target_tensor)

    def cross_entropy(self, p, p_target):
        return F.cross_entropy(p, p_target)

    def train(self, train_sample, writer):
        self.net.train()
        n = len(train_sample)
        state, probability, _, value = list(zip(*train_sample))
        state = torch.cat(state)
        state = state.cuda() if self.is_cuda else state
        probability = torch.tensor(np.array(probability), dtype=torch.float32)
        probability = probability.cuda() if self.is_cuda else probability
        value = torch.tensor(value, dtype=torch.float32).unsqueeze(1)
        value = value.cuda() if self.is_cuda else value

        batch_number = n // self.batch
        for epoch in range(self.epoch):
            print(f"Training {epoch}")
            for step in range(batch_number):
                start = step * self.batch
                state_training = state[start:start + self.batch, :, :, :]
                probability_training = probability[start:start + self.batch, :]
                value_training = value[start:start + self.batch]

                v_predict, p_predict = self.net(state_training)
                value_loss = self.smooth_l1(v_predict, value_training)
                probability_loss = self.cross_entropy(probability_training, p_predict)
                loss = value_loss + probability_loss
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()
                writer.add_float(loss.item(), "Loss")

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
