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
def self_play(nnet, num_sims, c_puct, temperature=1.0, dirichlet_alpha=0.3, max_steps=500):
    """
    使用 RMCTS 进行一盘自对弈，生成 (state, policy, z) 训练样本。
    加入最大步数限制、重复局面检测、超时返回棋子差。
    """
    state = game.rootState()
    history = []
    player = game.playerId(state)
    score = 0.0
    step = 0
    position_count = {}
    reason = 'n'
    while step < max_steps:
        step += 1
        actions = game.getValidActions(state)

        root = state[np.newaxis, :]
        pi, _ = learn_pi_and_v(root, num_sims, nnet, c_puct)
        pi = pi[0]

        if temperature == 1.0:
            noise = np.random.dirichlet([dirichlet_alpha] * len(actions))
            for i, a in enumerate(actions):
                pi[a] = 0.75 * pi[a] + 0.25 * noise[i]

        if temperature == 0:
            a = actions[np.argmax(pi[actions])]
        else:
            probs = pi[actions] ** (1.0 / temperature)
            probs /= np.sum(probs)
            a = np.random.choice(actions, p=probs)

        history.append((state.copy(), pi.copy(), player))
        state = game.nextState(state, a)
        player = game.playerId(state)

        # 检查游戏是否正常结束
        ended, score = game.gameEnded(state)
        if ended:
            print(f"Self-play 结束，得分 {score:.3f}")
            break

        # 重复局面检测（三次重复判负/按棋子差给分）
        s = ','.join(str(int(x)) for x in state.tolist())
        position_count[s] = position_count.get(s, 0) + 1
        if position_count[s] >= 3:
            # 对重复局面施加惩罚：在棋子差基础上减去 0.3（惩罚）
            score = get_dense_score(state) - 0.3
            # 确保分数在 [-1, 1] 范围内（clip）
            score = max(-1.0, min(1.0, score))
            print(f"重复局面，惩罚后得分 {score:.3f}")
            reason = 'r'
            break
    else:
        # 超时：按棋子差给分
        score = get_dense_score(state)
        print(f"Self-play 超时({step}步)，得分 {score:.3f}")

    # ========== 在 self_play 的末尾，所有循环结束后 ==========
    if abs(score) < 1e-6:  # 过滤 score=0 的数据
        print(f"⚠️ 丢弃平局数据 (步数 {len(history)})")
        return  # 直接返回，不 yield 任何数据

    # 生成训练样本（score 现在是连续值，不再是 0/-1/+1）
    z_abs = score
    for s, p, pl in history:
        z = z_abs * pl  # 转换到当前玩家视角
        yield s, p, z, reason

def get_dense_score(state):
    """
    根据当前棋盘状态计算归一化棋子差。
    范围约 [-1, 1]（21 个点，差值除以 21）。
    """
    board = state[1:]  # 跳过玩家 ID
    black = sum(1 for x in board if x == 1)
    white = sum(1 for x in board if x == -1)
    return (black - white) / 21.0

# ------------------------------------------------------------
# 2. 对战与评估函数
# ------------------------------------------------------------
def play_game(net1, net2, num_sims, c_puct, device, max_steps=500):
    """
    一局对战：net1 先手，net2 后手。
    返回终局得分（连续值，约 -1 ~ 1）。
    """
    state = game.rootState()
    player = game.playerId(state)
    step = 0
    position_count = {}
    score = 0.0

    while step < max_steps:
        step += 1
        actions = game.getValidActions(state)

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

        temperature_eval = 0.5
        probs = pi[actions] ** (1.0 / temperature_eval)
        probs /= np.sum(probs)
        best_action = np.random.choice(actions, p=probs)

        state = game.nextState(state, best_action)
        player = game.playerId(state)

        ended, score = game.gameEnded(state)
        if ended:
            print(f"对战 结束，得分 {score:.3f}")
            break

        s = ','.join(str(int(x)) for x in state.tolist())
        position_count[s] = position_count.get(s, 0) + 1
        if position_count[s] >= 3:
            score = get_dense_score(state)
            print(f"对战 重复局面，得分 {score:.3f}")
            break
    else:
        score = get_dense_score(state)
        print(f"对战 超时({step}步)，得分 {score:.3f}")

    return score

def evaluate(net, baseline_net, num_games, num_sims, c_puct, device, max_steps=500):
    """评估当前网络 vs 基线网络，返回胜率（当前网络先手胜率）"""
    wins = 0
    for i in range(num_games):
        result = play_game(net, baseline_net, num_sims, c_puct, device, max_steps=max_steps)
        print(f"{i} evaluate ended result : {result}")
        wins += result
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
    learning_rate = 0.0001
    eval_interval = 25          # 每20个epoch评估一次
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
        rep = 0
        for idx in range(num_selfplay_games):
            def nnet(states):
                with torch.no_grad():
                    states_t = torch.from_numpy(states).float().to(device)
                    logits, values = net(states_t)
                    probs = torch.softmax(logits, dim=1)
                return probs.cpu().numpy(), values.cpu().numpy().flatten()

            for state, policy, z, ren in self_play(nnet, num_sims, c_puct):
                replay_buffer.push(state, policy, z)
                if ren == 'r':
                    rep += 1

            print(f"{idx} Self-play done, buffer size: {len(replay_buffer)}")

        print(f"Epoch {epoch}, buffer size: {len(replay_buffer)}")

        # ---- 训练网络 ----
        if len(replay_buffer) >= batch_size:
            # 动态调整更新次数
            if len(replay_buffer) < 50000:
                num_updates = min(len(replay_buffer) // batch_size, 32)
            else:
                num_updates = min(len(replay_buffer) // batch_size, 32)

            # 用于累积指标
            total_loss = 0.0
            total_policy_loss = 0.0
            total_value_loss = 0.0
            total_pred_entropy = 0.0
            total_target_entropy = 0.0

            for _ in range(num_updates):
                states, target_policies, target_values = replay_buffer.sample(batch_size)

                states_t = torch.from_numpy(states).float().to(device)
                target_policies_t = torch.from_numpy(target_policies).float().to(device)
                target_values_t = torch.from_numpy(target_values).float().to(device).unsqueeze(1)

                logits, values = net(states_t)
                probs = torch.softmax(logits, dim=1)  # 预测策略
                log_probs = torch.log_softmax(logits, dim=1)

                policy_loss = -torch.mean(torch.sum(target_policies_t * log_probs, dim=1))
                value_loss = torch.mean((values - target_values_t) ** 2)

                # 3. 🔥 熵正则化（鼓励探索）
                # 计算当前预测策略的熵: -sum(p * log(p))
                entropy = -torch.mean(torch.sum(probs * log_probs, dim=1))
                # beta 是正则化系数，通常设一个很小的数，比如 0.01 或 0.005
                beta = 0.01
                entropy_loss = -beta * entropy  # 注意是减号，因为要让熵变大（即 loss 中减去熵）

                loss = policy_loss + value_loss + entropy_loss

                # 计算熵
                # 预测策略熵：-sum(p * log(p))
                pred_entropy = -torch.mean(torch.sum(probs * log_probs, dim=1)).item()
                # 目标策略熵：用同样的方式计算 target_policies 的熵（注意 target_policies 是概率分布）
                # 为了避免 log(0)，加一个小 epsilon
                target_log_probs = torch.log(target_policies_t + 1e-10)
                target_entropy = -torch.mean(torch.sum(target_policies_t * target_log_probs, dim=1)).item()

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                optimizer.step()

                total_loss += loss.item()
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_pred_entropy += pred_entropy
                total_target_entropy += target_entropy

            # 计算平均值
            avg_loss = total_loss / num_updates
            avg_policy_loss = total_policy_loss / num_updates
            avg_value_loss = total_value_loss / num_updates
            avg_pred_entropy = total_pred_entropy / num_updates
            avg_target_entropy = total_target_entropy / num_updates

            # 记录到 SwanLab
            sw.log({
                "重复局面": rep,
                "epoch": epoch,
                "avg_loss": avg_loss,
                "avg_policy_loss": avg_policy_loss,
                "avg_value_loss": avg_value_loss,
                "avg_pred_entropy": avg_pred_entropy,
                "avg_target_entropy": avg_target_entropy,
                "buffer_size": len(replay_buffer),
                "num_updates": num_updates,
            })

            if epoch % 10 == 0:
                print(
                    f"Epoch {epoch}, Avg Loss: {avg_loss:.4f}, Policy: {avg_policy_loss:.4f}, Value: {avg_value_loss:.4f}, PredEnt: {avg_pred_entropy:.3f}, TgtEnt: {avg_target_entropy:.3f}")
        # ---- 定期评估 ----
        if (epoch + 1) % eval_interval == 0:
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