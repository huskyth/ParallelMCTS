import torch
import torch.nn.functional as F
from torch.optim import Adam
from tqdm import tqdm
import numpy as np


class Wrapper:
    batch = 32

    def __init__(self, net):

        self.is_cuda = torch.cuda.is_available()
        self.net = net
        if self.is_cuda:
            self.net.cuda()
        self.opt = Adam(self.net.parameters(), lr=1e-3, weight_decay=1e-4)

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
    def mse(input_tensor, target_tensor):
        return torch.mean((input_tensor - target_tensor) ** 2)

    @staticmethod
    def cross(log_p, p_target):
        return -torch.mean(torch.sum(log_p * p_target, dim=-1))

    @torch.no_grad()
    def predict(self, state):
        v, p = self.net(state)
        return v.detach().cpu().numpy(), (torch.e ** p).detach().cpu().numpy()[0]

    def train_net(self, train_sample, swanlab):
        self.train()
        n = len(train_sample)
        epoch = 1
        print("🏠 Training epoch: ", epoch)
        batch_number = n // self.batch
        return_dict = []
        for epoch in range(epoch):
            value_loss_avg = 0
            probability_loss_avg = 0
            entropy_p_avg = 0
            t = tqdm(range(batch_number), desc='Training Net')
            for _ in t:
                sample_ids = np.random.randint(n, size=self.batch)
                s, p, _, r = list(zip(*[train_sample[i] for i in sample_ids]))

                boards = torch.FloatTensor(np.array(s, dtype=np.float64))
                target_pis = torch.FloatTensor(np.array(p))
                target_vs = torch.FloatTensor(np.array(r, dtype=np.float64))

                if self.is_cuda:
                    boards, target_pis, target_vs = boards.contiguous().cuda(), target_pis.contiguous().cuda(), target_vs.contiguous().cuda()

                out_v, out_pi = self.net(boards)
                l_pi = self.cross(out_pi, target_pis)
                l_v = self.mse(target_vs, out_v)
                total_loss = l_pi + l_v
                swanlab.log({
                    '策略损失': l_pi.item(), "价值损失": l_v.item(), "总损失": total_loss.item()
                })
                self.opt.zero_grad()
                total_loss.backward()
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
