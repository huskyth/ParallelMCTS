# train.py
import numpy as np
import torch
import torch.optim as optim
import swanlab as sw
import copy
import random
from . import game
from .RMCTS import learn_pi_and_v
from .wmnet import WatermelonNet
from .replay_buf import ReplayBuffer
from . import metaparm

sw.login(api_key="rdGaOSnlBY0KBDnNdkzja")
# ------------------------------------------------------------
# 1. 自对弈函数（生成训练数据）
# ------------------------------------------------------------
def self_play(nnet, num_sims, c_puct, temperature=1.0, dirichlet_alpha=0.3):
    """
    使用 RMCTS 进行一盘自对弈，生成 (state, policy, z) 训练样本。
    """
    state = game.rootState()
    history = []
    player = game.playerId(state)  # +1 或 -1

    while True:
        actions = game.getValidActions(state)
        if len(actions) == 0:
            break

        # 搜索
        root = state[np.newaxis, :]
        pi, _ = learn_pi_and_v(root, num_sims, nnet, c_puct)
        pi = pi[0]  # 去掉 batch 维度

        # 添加 Dirichlet 噪声
        if temperature == 1.0:
            noise = np.random.dirichlet([dirichlet_alpha] * len(actions))
            for i, a in enumerate(actions):
                pi[a] = 0.75 * pi[a] + 0.25 * noise[i]

        # 采样动作
        if temperature == 0:
            a = actions[np.argmax(pi[actions])]
        else:
            probs = pi[actions] ** (1.0 / temperature)
            probs /= np.sum(probs)
            a = np.random.choice(actions, p=probs)

        # 记录
        history.append((state.copy(), pi.copy(), player))

        # 执行动作
        state = game.nextState(state, a)
        player = game.playerId(state)

        # 检查终局
        ended, score = game.gameEnded(state)
        if ended:
            break

    # 生成训练样本
    z_abs = score  # 相对于玩家1
    for s, p, pl in history:
        z = z_abs * pl
        yield s, p, z


# ------------------------------------------------------------
# 2. 对战与评估函数
# ------------------------------------------------------------
def play_game(net1, net2, num_sims, c_puct, device):
    """
    一局对战：net1 先手，net2 后手。
    返回终局得分（+1 表示 net1 赢，-1 表示 net2 赢，0 平局）。
    """
    state = game.rootState()
    player = game.playerId(state)
    while True:
        actions = game.getValidActions(state)
        if len(actions) == 0:
            break
        # 选择当前玩家对应的网络
        net = net1 if player == 1 else net2

        def nnet(states):
            with torch.no_grad():
                states_t = torch.from_numpy(states).float().to(device)
                logits, values = net(states_t)
                probs = torch.softmax(logits, dim=1)
            return probs.cpu().numpy(), values.cpu().numpy().flatten()

        root = state[np.newaxis, :]
        pi, _ = learn_pi_and_v(root, num_sims, nnet, c_puct)
        pi = pi[0]
        # 确定性选择最佳动作
        best_action = actions[np.argmax(pi[actions])]
        state = game.nextState(state, best_action)
        player = game.playerId(state)

        ended, score = game.gameEnded(state)
        if ended:
            break
    return score  # net1 作为玩家1，得分 +1 表示胜


def evaluate(net, baseline_net, num_games, num_sims, c_puct, device):
    """评估当前网络 vs 基线网络，返回胜率（当前网络先手胜率）"""
    wins = 0
    for _ in range(num_games):
        result = play_game(net, baseline_net, num_sims, c_puct, device)
        print(f"Play Done result {result}")
        if result == 1:
            wins += 1
    return wins / num_games


# ------------------------------------------------------------
# 3. 主训练循环
# ------------------------------------------------------------
def train():
    # ---- 超参数 ----
    num_sims = 400
    c_puct = metaparm.c_puct
    batch_size = 256
    num_selfplay_games = 32
    num_epochs = 1000
    learning_rate = 0.001
    eval_interval = 20          # 每20个epoch评估一次
    eval_games = 50             # 每评估50盘
    eval_sims = 200             # 评估时搜索次数（可小于训练值）
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
        reinit=True
    )

    # ---- 网络、优化器、回放池 ----
    net = WatermelonNet(input_dim=game.gameLength(), num_actions=game.numActions()).to(device)
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    replay_buffer = ReplayBuffer(max_size=200000)

    # ---- 保存基线模型（初始网络） ----
    baseline_net = copy.deepcopy(net)

    # ---- 训练循环 ----
    for epoch in range(num_epochs):
        # ---- 自对弈生成数据 ----
        for idx in range(num_selfplay_games):
            def nnet(states):
                with torch.no_grad():
                    states_t = torch.from_numpy(states).float().to(device)
                    logits, values = net(states_t)
                    probs = torch.softmax(logits, dim=1)
                return probs.cpu().numpy(), values.cpu().numpy().flatten()

            for state, policy, z in self_play(nnet, num_sims, c_puct):
                replay_buffer.push(state, policy, z)

            print(f"{idx} Self-play done, buffer size: {len(replay_buffer)}")

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

            # 记录训练损失
            sw.log({
                "epoch": epoch,
                "loss": loss.item(),
                "policy_loss": policy_loss.item(),
                "value_loss": value_loss.item(),
                "buffer_size": len(replay_buffer),
            })

            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {loss.item():.4f}, Policy Loss: {policy_loss.item():.4f}, Value Loss: {value_loss.item():.4f}")

        # ---- 定期评估 ----
        if epoch % eval_interval == 0:
            win_rate = evaluate(net, baseline_net, eval_games, eval_sims, c_puct, device)
            print(f"Epoch {epoch}, Win Rate vs Baseline: {win_rate:.3f}")
            sw.log({"win_rate_vs_baseline": win_rate, "epoch": epoch})

        # ---- 保存模型 ----
        if epoch % 50 == 0:
            model_path = f"model_epoch_{epoch}.pth"
            torch.save(net.state_dict(), model_path)
            sw.save(model_path)

    sw.finish()


if __name__ == "__main__":
    train()