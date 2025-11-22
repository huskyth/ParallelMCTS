import torch
from constants import ROOT_PATH
from models.models import GameNet
from models.wrapper import Wrapper


class ChessNetWrapper(Wrapper):
    MODEL_SAVE_PATH = ROOT_PATH / "checkpoints" / "WMChess"
    if not MODEL_SAVE_PATH.exists():
        MODEL_SAVE_PATH.mkdir()

    def __init__(self):
        self.net = GameNet(2, 7, 72)
        super().__init__(self.net)

    def train(self, train_sample):
        n = len(train_sample)
        state, probability, _, value = list(zip(*train_sample))

        state = torch.stack(state).float()
        state = state.cuda() if self.is_cuda else state

        probability = torch.tensor(probability).float()
        probability = probability.cuda() if self.is_cuda else probability

        value = torch.stack(value)[:, None].float()
        value = value.cuda() if self.is_cuda else value

        if state.shape != (n, 7, 7, 2) or probability.shape != (n, 72) or value.shape != (n, 1):
            raise ValueError(
                f"state, probability, value shape error, shape is {state.shape}, {probability.shape}, {value.shape}")

        epoch = min(max(n // 100 + 1, 3), 10)
        print("🏠 Training epoch: ", epoch)
        batch_number = n // self.batch
        return_dict = []
        for epoch in range(epoch):
            value_loss_avg = 0
            probability_loss_avg = 0
            entropy_p_avg = 0
            for step in range(batch_number):
                start = step * self.batch
                state_batch = state[start:start + self.batch, :, :, :]
                probability_batch = probability[start:start + self.batch, :]
                value_batch = value[start:start + self.batch]
                v_predict, p_predict = self.net(state_batch)
                value_loss = self.smooth_l1(v_predict, value_batch)

                target = torch.argmax(probability_batch, dim=1)

                probability_loss = self.cross_entropy(torch.e ** p_predict, target)

                entropy_p = (-torch.e ** p_predict * p_predict).sum(axis=1).mean().item()

                value_loss_avg = (value_loss_avg * step + value_loss.item()) / (1 + step)
                probability_loss_avg = (probability_loss_avg * step + probability_loss.item()) / (1 + step)
                entropy_p_avg = (entropy_p_avg * step + entropy_p) / (1 + step)

                if torch.isclose(torch.tensor(entropy_p), torch.tensor(0.0)):
                    print(f"🐰 为什么熵为0，看看张量：{p_predict}")

                loss = value_loss + probability_loss

                self.opt.zero_grad()
                loss.backward()
                self.opt.step()

            return_dict.append({
                "value_loss_avg": value_loss_avg, "probability_loss_avg": probability_loss_avg, "entropy_p_avg":
                    entropy_p_avg
            })

        return return_dict
