import torch
import torch.optim as optim

import sys
from . import game     # 通过包名导入

from . import replay_buf
from . import self_play
from .RMCTS import MCTS

from .wmnet import WatermelonNet


def train():
    # 超参数
    num_sims = 400
    c_puct = 1.0
    batch_size = 256
    num_selfplay_games = 32   # 每轮生成多少盘棋
    num_epochs = 1000
    buffer_size = 200000
    dirichlet_alpha = 0.3
    temperature = 1.0

    # 设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 网络
    net = WatermelonNet(input_dim=game.gameLength(), num_actions=game.numActions()).to(device)
    optimizer = optim.Adam(net.parameters(), lr=0.001)

    # 经验池
    replay_buffer = replay_buf.ReplayBuffer(max_size=buffer_size)

    # 记录
    for epoch in range(num_epochs):
        # ---- 自对弈生成数据 ----
        for _ in range(num_selfplay_games):
            # 为每盘棋创建一个新的 MCTS 实例（因为 MCTS 内部状态与搜索树绑定）
            # 但我们可以复用同一个 net，每次传入不同的状态
            # 这里我们直接用 self_play 函数，它内部会创建 MCTS
            # 为了效率，可以在外部创建 MCTS，但每次搜索会重置树，所以每次都要新建。
            mcts = MCTS(None, num_sims, c_puct, None)  # 先传 None，后面在 self_play 中会传入网络
            # 但是 self_play 里需要 net 来推理，所以我们要传入 net 的推理函数
            def nnet(states):
                # 将 numpy 数组转为 torch tensor，batch 推理
                states_t = torch.from_numpy(states).float().to(device)
                with torch.no_grad():
                    logits, values = net(states_t)
                    probs = torch.softmax(logits, dim=1)
                return probs.cpu().numpy(), values.cpu().numpy().flatten()

            # 自对弈生成数据
            for state, policy, z in self_play(mcts, game, nnet, temperature, dirichlet_alpha):
                replay_buffer.push(state, policy, z)

        print(f"Epoch {epoch}, buffer size: {len(replay_buffer)}")

        # ---- 训练网络 ----
        if len(replay_buffer) >= batch_size:
            states, target_policies, target_values = replay_buffer.sample(batch_size)
            states_t = torch.from_numpy(states).float().to(device)
            target_policies_t = torch.from_numpy(target_policies).float().to(device)
            target_values_t = torch.from_numpy(target_values).float().to(device)

            optimizer.zero_grad()
            logits, values = net(states_t)
            # 策略损失（交叉熵）
            policy_loss = -torch.mean(torch.sum(target_policies_t * torch.log_softmax(logits, dim=1), dim=1))
            # 价值损失（MSE）
            value_loss = torch.mean((values - target_values_t) ** 2)
            loss = policy_loss + value_loss
            loss.backward()
            optimizer.step()

            print(f"  Loss: {loss.item():.4f}, Policy loss: {policy_loss.item():.4f}, Value loss: {value_loss.item():.4f}")

        # 保存模型
        if epoch % 50 == 0:
            torch.save(net.state_dict(), f"model_epoch_{epoch}.pth")

if __name__ == "__main__":
    train()