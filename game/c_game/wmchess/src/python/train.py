# train.py
import numpy as np
import torch
import torch.optim as optim
from . import game
from .self_play import self_play
from .wmnet import WatermelonNet
from .replay_buf import ReplayBuffer
from . import metaparm   # 包含 c_puct 等

def train():
    num_sims = 400
    c_puct = metaparm.c_puct
    batch_size = 256
    num_selfplay_games = 32
    num_epochs = 1000
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    net = WatermelonNet(input_dim=game.gameLength(), num_actions=game.numActions()).to(device)
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    replay_buffer = ReplayBuffer(max_size=200000)

    for epoch in range(num_epochs):
        # ---- 自对弈生成数据 ----
        for _ in range(num_selfplay_games):
            # 定义网络推理函数（将 PyTorch 模型封装为 numpy 接口）
            def nnet(states):
                # states: numpy array (batch, gameLength)
                with torch.no_grad():
                    states_t = torch.from_numpy(states).float().to(device)
                    logits, values = net(states_t)
                    probs = torch.softmax(logits, dim=1)
                return probs.cpu().numpy(), values.cpu().numpy().flatten()

            # 调用 self_play，内部使用 learn_pi_and_v
            for state, policy, z in self_play(nnet, num_sims, c_puct):
                replay_buffer.push(state, policy, z)

        print(f"Epoch {epoch}, buffer size: {len(replay_buffer)}")

        # ---- 训练网络 ----
        if len(replay_buffer) >= batch_size:
            # 采样、训练...
            pass

        # 保存模型
        if epoch % 50 == 0:
            torch.save(net.state_dict(), f"model_epoch_{epoch}.pth")