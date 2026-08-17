# train.py
import json
import time

import numpy as np
import torch
import torch.optim as optim
import swanlab as sw
import copy
import random
import ctypes
import os
import pickle  # 新增
from . import game
from .RMCTS import learn_pi_and_v
from .wmnet_gcn import WatermelonGCN
from .replay_buf import ReplayBuffer
from . import metaparm

sw.login(api_key="rdGaOSnlBY0KBDnNdkzja")

SAVE_TRAJECTORY = True          # True 表示保存每局轨迹到磁盘
TRAJECTORY_DIR = "./trajectories"  # 保存目录
# ------------------------------------------------------------
# 1. 自对弈函数（只保留自然终局）
# ------------------------------------------------------------
def self_play(nnet, num_sims, c_puct, temperature=1.0, dirichlet_alpha=0.3, max_steps=500):
    state = game.rootState()
    history = []
    player = game.playerId(state)
    step = 0
    position_count = {}
    reason = 'n'
    gamma = 0.95
    step_rewards = []
    terminal_score = 0.0

    while step < max_steps:
        step += 1
        actions = game.getValidActions(state)

        root = state[np.newaxis, :]
        pi, _ = learn_pi_and_v(root, num_sims, nnet, c_puct)
        pi = pi[0]

        pure_pi = pi.copy()
        legal_pi = pi[actions]

        if temperature == 1.0:
            noise = np.random.dirichlet([dirichlet_alpha] * len(actions))
            legal_pi = 0.75 * legal_pi + 0.25 * noise

        if temperature == 0:
            a = actions[np.argmax(legal_pi)]
        else:
            scaled_probs = legal_pi ** (1.0 / temperature)
            scaled_probs /= np.sum(scaled_probs)
            a = np.random.choice(actions, p=scaled_probs)

        history.append((state.copy(), pure_pi.copy(), player))

        captures = ctypes.c_int()
        state = game.nextState(state, a, captures)
        r = captures.value * 0.3
        step_rewards.append(r)

        player = game.playerId(state)

        ended, score = game.gameEnded(state)
        if ended:
            terminal_score = score
            reason = 'e'
            break

        s = ','.join(str(int(x)) for x in state.tolist())
        position_count[s] = position_count.get(s, 0) + 1
        if position_count[s] >= 3:
            reason = 'r'
            break
    else:
        reason = 't'

    KEEP_STEPS = 60
    if len(history) > KEEP_STEPS:
        history = history[-KEEP_STEPS:]
        step_rewards = step_rewards[-KEEP_STEPS:]

    if reason == 'e':
        cumulative = terminal_score
    else:
        cumulative = get_dense_score(state)

    returns = []
    for r in reversed(step_rewards):
        cumulative = r + gamma * cumulative
        returns.append(cumulative)
    returns.reverse()

    # ========== 🔥 新增：保存完整轨迹到磁盘 ==========
    if SAVE_TRAJECTORY:
        # 创建目录
        os.makedirs(TRAJECTORY_DIR, exist_ok=True)
        # 生成时间戳文件名
        timestamp = int(time.time() * 1000)
        filename = f"{TRAJECTORY_DIR}/traj_{timestamp}.json"

        # 构造可序列化的轨迹数据
        traj_data = {
            "reason": reason,
            "terminal_score": terminal_score,
            "step_rewards": [float(r) for r in step_rewards],
            "returns": [float(r) for r in returns],
            "history": []
        }
        for i, (s, p, pl) in enumerate(history):
            traj_data["history"].append({
                "state": s.tolist(),  # 棋盘状态（22维）
                "policy": p.tolist(),  # 策略分布（72维）
                "player": int(pl),
                "step_reward": float(step_rewards[i]) if i < len(step_rewards) else 0.0,
                "return": float(returns[i]) if i < len(returns) else 0.0,
            })
        # 写入文件
        with open(filename, "w") as f:
            json.dump(traj_data, f, indent=2)
        # 可选：打印提示（但可能干扰训练输出，可注释掉）
        # print(f"💾 轨迹已保存到 {filename}")

    # 生成训练样本（原有逻辑）
    for i, (s, p, pl) in enumerate(history):
        z = returns[i] * pl
        yield s, p, z, reason

def get_dense_score(state):
    board = state[1:]
    black = sum(1 for x in board if x == 1)
    white = sum(1 for x in board if x == -1)
    return (black - white) / 21.0


# ------------------------------------------------------------
# 2. 确定性对战（用于评估）
# ------------------------------------------------------------
def play_game_deterministic(net1, net2, num_sims, c_puct, device, state=None, max_steps=300):
    """
    纯确定性评估：永远走 argmax，无随机采样。
    如果 state 为 None，则使用默认开局。
    """
    if state is None:
        state = game.rootState()
    else:
        state = state.copy()

    player = game.playerId(state)
    step = 0
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

        # 纯 argmax（无随机）
        best_action = actions[np.argmax(pi[actions])]

        state, _ = game.nextState(state, best_action)
        player = game.playerId(state)

        ended, score = game.gameEnded(state)
        if ended:
            break
    else:
        # 超时，返回当前棋子差
        score = get_dense_score(state)

    return score


# ------------------------------------------------------------
# 3. 评估函数：当前网络 vs 纯随机（无搜索）
# ------------------------------------------------------------
def evaluate_vs_pure_random(net, num_sims, c_puct, device, num_starts=5, max_steps=300):
    """
    在多个不同的初始状态下，让当前网络（带搜索）对阵纯随机走子（无搜索）。
    返回平均得分（>0 表示网络占优）。
    """
    np.random.seed(42)  # 固定种子确保可复现
    scores = []

    for _ in range(num_starts):
        # 随机生成初始状态（从默认开局走 5~15 步）
        state = game.rootState()

        # 在该状态下进行对决
        player = game.playerId(state)
        step = 0
        score = 0.0

        while step < max_steps:
            step += 1
            actions = game.getValidActions(state)

            if player == 1:
                # 当前网络走棋（带 RMCTS）
                def nnet(states):
                    with torch.no_grad():
                        states_t = torch.from_numpy(states).float().to(device)
                        logits, values = net(states_t)
                        probs = torch.softmax(logits, dim=1)
                    return probs.cpu().numpy(), values.cpu().numpy().flatten()

                root = state[np.newaxis, :]
                pi, _ = learn_pi_and_v(root, num_sims, nnet, c_puct)
                pi = pi[0]
                # 评估温度设为 0.3（既能保持网络偏好，又给一点随机性）
                temperature_eval = 0.3
                probs = pi[actions] ** (1.0 / temperature_eval)
                probs /= np.sum(probs)
                best_action = np.random.choice(actions, p=probs)
            else:
                # 对手：纯随机走子（无搜索）
                best_action = np.random.choice(actions)

            state, _ = game.nextState(state, best_action)
            player = game.playerId(state)

            ended, score = game.gameEnded(state)
            if ended:
                break
        else:
            score = get_dense_score(state)

        scores.append(score)

    return np.mean(scores)


# ------------------------------------------------------------
# 4. 辅助评估：与固定随机基线比较（确定性）
# ------------------------------------------------------------
def evaluate_deterministic_avg(net, baseline_net, num_sims, c_puct, device, num_starts=5):
    """
    在多个不同初始状态下，用确定性走法比较 net 与 baseline_net，返回平均得分。
    """
    np.random.seed(42)
    scores = []
    for _ in range(num_starts):
        state = game.rootState()
        steps = np.random.randint(5, 15)
        for _ in range(steps):
            actions = game.getValidActions(state)
            a = np.random.choice(actions)
            state, _ = game.nextState(state, a)
            ended, _ = game.gameEnded(state)
            if ended:
                break
        score = play_game_deterministic(net, baseline_net, num_sims, c_puct, device, state)
        scores.append(score)
    return np.mean(scores)


# ------------------------------------------------------------
# 5. 主训练循环
# ------------------------------------------------------------
def train():
    # ---- 超参数 ----
    num_sims = 400
    c_puct = metaparm.c_puct
    batch_size = 256
    num_selfplay_games = 3
    num_epochs = 1000
    learning_rate = 0.0001
    eval_interval = 3
    num_starts = 10  # 评估使用的初始状态数
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    best_model_path = "best_model.pth"
    buffer_path = "replay_buffer.pkl"

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
    net = WatermelonGCN().to(device)

    start_epoch = 0
    if os.path.exists(best_model_path):
        print(f"📂 发现已有 best_model.pth，加载权重继续训练...")
        net.load_state_dict(torch.load(best_model_path, map_location=device))
    else:
        print("🆕 未找到 best_model.pth，从随机初始化开始训练。")

    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    replay_buffer = ReplayBuffer(max_size=200000)

    # ---- 固定随机基线（仅用于参考，不参与决策） ----
    random_net = WatermelonGCN().to(device)
    baseline_net = copy.deepcopy(random_net)  # 固定，不更新


    # ---- 训练循环 ----
    for epoch in range(start_epoch, num_epochs):

        for idx in range(num_selfplay_games):
            def nnet(states):
                with torch.no_grad():
                    states_t = torch.from_numpy(states).float().to(device)
                    logits, values = net(states_t)
                    probs = torch.softmax(logits, dim=1)
                return probs.cpu().numpy(), values.cpu().numpy().flatten()

            rep = 0
            for state, policy, z, ren in self_play(nnet, num_sims, c_puct):
                replay_buffer.push(state, policy, z)
                rep += 1
            sw.log({"self_play长": rep, "epoch": epoch})
            print(f"{idx} Self-play done, buffer size: {len(replay_buffer)}")

        print(f"Epoch {epoch}, buffer size: {len(replay_buffer)}")

        # ---- 训练网络 ----
        if len(replay_buffer) >= batch_size:
            num_updates = min(len(replay_buffer) // batch_size, 8)

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
                probs = torch.softmax(logits, dim=1)
                log_probs = torch.log_softmax(logits, dim=1)

                policy_loss = -torch.mean(torch.sum(target_policies_t * log_probs, dim=1))
                value_loss = torch.mean((values - target_values_t) ** 2)

                entropy = -torch.mean(torch.sum(probs * log_probs, dim=1))
                beta = 0.02
                entropy_loss = -beta * entropy
                pw = 0.8
                vw = 3
                loss = pw * policy_loss + vw * value_loss + entropy_loss

                pred_entropy = -torch.mean(torch.sum(probs * log_probs, dim=1)).item()
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

            avg_loss = total_loss / num_updates
            avg_policy_loss = total_policy_loss / num_updates
            avg_value_loss = total_value_loss / num_updates
            avg_pred_entropy = total_pred_entropy / num_updates
            avg_target_entropy = total_target_entropy / num_updates

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

        # ---- 评估 ----
        if (epoch + 1) % eval_interval == 0:
            # 主评估：当前网络 vs 纯随机（无搜索）
            avg_score_vs_random = evaluate_vs_pure_random(
                net, num_sims, c_puct, device, num_starts=num_starts
            )
            print(f"Epoch {epoch}, Avg Score vs Pure Random: {avg_score_vs_random:.4f}")
            sw.log({"avg_score_vs_pure_random": avg_score_vs_random, "epoch": epoch})

            # 更新教师模型：若对纯随机胜率（得分）高于 0.1，认为有明显优势
            if avg_score_vs_random > 0.1:
                torch.save(net.state_dict(), best_model_path)
                sw.save(best_model_path)
                print(f"🏆 更新历史最优模型！Epoch {epoch}, 得分 {avg_score_vs_random:.4f}")

        # ---- 保存模型 ----
        if epoch % 50 == 0:
            model_path = f"model_epoch_{epoch}.pth"
            torch.save(net.state_dict(), model_path)
            sw.save(model_path)

    sw.finish()
    print(f"\n🎉 训练完成！最终最佳模型保存在 {best_model_path}")


if __name__ == "__main__":
    train()
