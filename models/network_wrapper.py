import torch
import torch.nn.functional as F
from torch.optim import Adam

from constants import ROOT_PATH
from models.models import ChessNet

MODEL_SAVE_PATH = ROOT_PATH / "checkpoints"
if not MODEL_SAVE_PATH.exists():
    MODEL_SAVE_PATH.mkdir()


class ChessNetWrapper:
    def __init__(self, swlab):
        self.net = ChessNet()

        self.is_cuda = torch.cuda.is_available()
        self.opt = Adam(self.net.parameters(), lr=1e-3, weight_decay=1e-2)

        self.epoch = 10
        self.batch = 1

        self.swlab = swlab

    @torch.no_grad()
    def predict(self, state):
        v, p = self.net(state)
        return v.detach().cpu().numpy(), p.detach().cpu().numpy()[0]

    def smooth_l1(self, input_tensor, target_tensor):
        return F.smooth_l1_loss(input_tensor, target_tensor)

    def cross_entropy(self, p, p_target):
        return F.cross_entropy(p, p_target)

    def train(self, train_sample):
        n = len(train_sample)
        state, probability, _, value = list(zip(*train_sample))

        state = torch.stack(state).float()
        state = state.cuda() if self.is_cuda else state
        state = state.unsqueeze(1)

        probability = torch.tensor(probability).float()
        probability = probability.cuda() if self.is_cuda else probability

        value = torch.stack(value)[:, None].float()
        value = value.cuda() if self.is_cuda else value

        if state.shape != (n, 7, 7, 2) or probability.shape != (n, 72) or value.shape != (n, 1):
            raise ValueError("state, probability, value shape error")

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

                entropy_p = (-torch.e ** p_predict * p_predict).sum(axis=1).mean().item().detach().cpu()

                self.swlab.log(
                    {"value_loss": value_loss.item(), "probability_loss": probability_loss.item(), "entropy_p":
                        entropy_p})

                loss = value_loss + probability_loss

                self.opt.zero_grad()
                loss.backward()
                self.opt.step()

    def save(self, epoch, key="latest.pt"):
        checkpoint = {
            'epoch': epoch,
            'state_dict': self.net.state_dict(),
            'optimizer': self.opt.state_dict(),
        }
        torch.save(checkpoint, str(MODEL_SAVE_PATH / key))
        print(f"{key} 模型已经保存")

    def load(self, key):
        model = torch.load(str(MODEL_SAVE_PATH / key))
        self.net.load_state_dict(model["state_dict"])
        self.opt.load_state_dict(model["optimizer"])
        print(f"{key} 模型已经加载")
        return model["epoch"]
