import random
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
        self.batch = 512

    @torch.no_grad()
    def predict(self, state):
        self.net.eval()
        if len(state.shape) == 2:
            state = state.unsqueeze(0).unsqueeze(0)
        state = state.cuda() if self.is_cuda else state
        v, p = self.net(state)
        return v.detach().cpu().numpy(), p.exp().detach().cpu().numpy()

    def mse(self, input_tensor, target_tensor):
        return F.mse_loss(input_tensor, target_tensor)

    def cross_entropy(self, p_target, predict):
        return -torch.sum(p_target * predict) / p_target.size()[0]

    def train(self, train_sample, writer, epoch_numbers, batch_size):
        train_sample = random.sample(train_sample, batch_size)
        self.net.train()
        n = len(train_sample)
        state, probability, _, value = list(zip(*train_sample))
        state = torch.cat(state)
        state = state.cuda() if self.is_cuda else state
        probability = torch.tensor(np.array(probability), dtype=torch.float32)
        probability = probability.cuda() if self.is_cuda else probability
        value = torch.tensor(value, dtype=torch.float32).unsqueeze(1)
        value = value.cuda() if self.is_cuda else value

        writer.add_float(epoch_numbers, "Training Times")
        for epoch in range(epoch_numbers):
            print(f"Training {epoch}")
            state_training = state
            probability_training = probability
            value_training = value

            v_predict, p_predict = self.net(state_training)
            value_loss = self.mse(v_predict, value_training)
            probability_loss = self.cross_entropy(probability_training, p_predict)
            loss = value_loss + probability_loss
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
            writer.add_float(value_loss.item(), "Value Loss")
            writer.add_float(probability_loss.item(), "Probability Loss")
            writer.add_float(loss.item(), "Loss")

            _, p_inference = self.predict(state_training)
            select_move_predict = np.argmax(p_inference, axis=1)
            select_move_target = np.argmax(probability_training.cpu().numpy(), axis=1)
            success_rate = (select_move_target == select_move_predict).sum().item() / len(probability_training)
            writer.add_float(success_rate, "Success Rate")

    def save(self, key):
        torch.save({"state_dict": self.net.state_dict()}, str(MODEL_SAVE_PATH / key))

    def load(self, key):
        model = torch.load(str(MODEL_SAVE_PATH / key))
        self.net.load_state_dict(model["state_dict"])


if __name__ == '__main__':
    cn = ChessNetWrapper()
    cn.cross_entropy(torch.randn(8, 72), torch.randn((8, 72)))
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
