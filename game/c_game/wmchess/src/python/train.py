# train.py
import numpy as np
import torch
import torch.optim as optim
import swanlab as sw
from . import game
from .self_play import self_play
from .wmnet import WatermelonNet
from .replay_buf import ReplayBuffer
from . import metaparm

sw.login(api_key="rdGaOSnlBY0KBDnNdkzja")


def train():
    # ---- 超参数 ----
    num_sims = 400
    c_puct = metaparm.c_puct
    batch_size = 256
    num_selfplay_games = 32
    num_epochs = 1000
    learning_rate = 0.001
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- 初始化 SwanLab ----
    sw.init(
        project="watermelon-chess-rmcts",
        config={
            "num_sims": num_sims,
            "c_puct": c_puct,
            "batch_size": batch_size,
            "num_selfplay_games": num_selfplay_games,
            "num_epochs": num_epochs,
            "learning_rate": learning_rate,
        },
        reinit=True  # 允许重复初始化
    )

    net = WatermelonNet(input_dim=game.gameLength(), num_actions=game.numActions()).to(device)
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    replay_buffer = ReplayBuffer(max_size=200000)

    for epoch in range(num_epochs):
        # ---- 自对弈生成数据 ----
        for _ in range(num_selfplay_games):
            def nnet(states):
                with torch.no_grad():
                    states_t = torch.from_numpy(states).float().to(device)
                    logits, values = net(states_t)
                    probs = torch.softmax(logits, dim=1)
                return probs.cpu().numpy(), values.cpu().numpy().flatten()

            for state, policy, z in self_play(nnet, num_sims, c_puct):
                replay_buffer.push(state, policy, z)

        print(f"Epoch {epoch}, buffer size: {len(replay_buffer)}")

        # ---- 训练网络 ----
        if len(replay_buffer) >= batch_size:
            states, target_policies, target_values = replay_buffer.sample(batch_size)

            states_t = torch.from_numpy(states).float().to(device)
            target_policies_t = torch.from_numpy(target_policies).float().to(device)
            target_values_t = torch.from_numpy(target_values).float().to(device).unsqueeze(1)

            logits, values = net(states_t)
            log_probs = torch.log_softmax(logits, dim=1)
            policy_loss = -torch.mean(torch.sum(target_policies_t * log_probs, dim=1))
            value_loss = torch.mean((values - target_values_t) ** 2)
            loss = policy_loss + value_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # ---- 记录到 SwanLab ----
            sw.log({
                "epoch": epoch,
                "loss": loss.item(),
                "policy_loss": policy_loss.item(),
                "value_loss": value_loss.item(),
                "buffer_size": len(replay_buffer),
            })

            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {loss.item():.4f}, Policy Loss: {policy_loss.item():.4f}, Value Loss: {value_loss.item():.4f}")

        # ---- 保存模型 ----
        if epoch % 50 == 0:
            model_path = f"model_epoch_{epoch}.pth"
            torch.save(net.state_dict(), model_path)
            sw.save(model_path)

    sw.finish()

if __name__ == "__main__":
    train()