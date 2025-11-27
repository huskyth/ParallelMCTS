import torch
import torch.nn.functional as F
from torch.optim import Adam


class Wrapper:
    batch = 4

    def __init__(self, net):

        self.is_cuda = torch.cuda.is_available()
        self.net = net
        if self.is_cuda:
            self.net.cuda()
        self.opt = Adam(self.net.parameters(), lr=1e-4, weight_decay=1e-2)

    def save(self, epoch, key="latest.pt"):
        checkpoint = {
            'epoch': epoch,
            'state_dict': self.net.state_dict(),
            'optimizer': self.opt.state_dict(),
        }
        torch.save(checkpoint, str(self.MODEL_SAVE_PATH / key))
        print(f"{key} 模型已经保存")

    def load(self, key):
        model = torch.load(str(self.MODEL_SAVE_PATH / key))
        self.net.load_state_dict(model["state_dict"])
        self.opt.load_state_dict(model["optimizer"])
        print(f"{key} 模型已经加载")
        return model["epoch"]

    def try_load(self):
        try:
            epoch = self.load("latest.pt")
        except Exception as e:
            print(e)
            epoch = 0

        return epoch

    @staticmethod
    def smooth_l1(input_tensor, target_tensor):
        return F.smooth_l1_loss(input_tensor, target_tensor)

    @staticmethod
    def cross_entropy(p, p_target):
        return F.cross_entropy(p, p_target)

    @torch.no_grad()
    def predict(self, state):
        v, p = self.net(state)
        return v.detach().cpu().numpy(), (torch.e ** p).detach().cpu().numpy()[0]

    def train_net(self, train_sample):
        self.train()
        n = len(train_sample)
        state, probability, _, value = list(zip(*train_sample))

        state = torch.stack(state).float()
        state = state.cuda() if self.is_cuda else state

        probability = torch.tensor(probability).float()
        probability = probability.cuda() if self.is_cuda else probability

        value = torch.stack(value)[:, None].float()
        value = value.cuda() if self.is_cuda else value

        if state.shape != (n, self.net.input_size, self.net.input_size, 3) or probability.shape != (
                n, self.net.action_size) or value.shape != (n, 1):
            raise ValueError(
                f"state, probability, value shape error, shape is {state.shape}, {probability.shape}, {value.shape}")

        epoch = min(max(n // 100 + 1, 3), 5)
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

    def eval(self):
        self.net.eval()

    def train(self):
        self.net.train()
