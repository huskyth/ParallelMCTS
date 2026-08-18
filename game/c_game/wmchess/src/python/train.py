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
import pickle
from . import game
from .RMCTS import learn_pi_and_v
from .wmnet_gcn import WatermelonGCN
from .replay_buf import ReplayBuffer
from . import metaparm

sw.login(api_key="rdGaOSnlBY0KBDnNdkzja")

SAVE_TRAJECTORY = False
TRAJECTORY_DIR = "./trajectories"

# ------------------------------------------------------------
# 1. 自对弈函数
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

    # KEEP_STEPS = 60
    # if len(history) > KEEP_STEPS:
    #     history = history[-KEEP_STEPS:]
    #     step_rewards = step_rewards[-KEEP_STEPS:]

    if reason == 'e':
        cumulative = terminal_score
    else:
        cumulative = get_dense_score(state)

    returns = []
    for r in reversed(step_rewards):
        cumulative = r + gamma * cumulative
        returns.append(cumulative)
    returns.reverse()

    if SAVE_TRAJECTORY:
        os.makedirs(TRAJECTORY_DIR, exist_ok=True)
        timestamp = int(time.time() * 1000)
        filename = f"{TRAJECTORY_DIR}/traj_{timestamp}.json"
        traj_data = {
            "reason": reason,
            "terminal_score": terminal_score,
            "step_rewards": [float(r) for r in step_rewards],
            "returns": [float(r) for r in returns],
            "history": []
        }
        for i, (s, p, pl) in enumerate(history):
            traj_data["history"].append({
                "state": s.tolist(),
                "policy": p.tolist(),
                "player": int(pl),
                "step_reward": float(step_rewards[i]) if i < len(step_rewards) else 0.0,
                "return": float(returns[i]) if i < len(returns) else 0.0,
            })
        with open(filename, "w") as f:
            json.dump(traj_data, f, indent=2)

    for i, (s, p, pl) in enumerate(history):
        z = returns[i] * pl
        yield s, p, z, reason


def get_dense_score(state):
    board = state[1:]
    black = sum(1 for x in board if x == 1)
    white = sum(1 for x in board if x == -1)
    return (black - white) / 21.0


# ------------------------------------------------------------
# 2. 对战与评估函数
# ------------------------------------------------------------
def play_game(net1, net2, num_sims, c_puct, device, state=None, max_steps=300):
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

        temperature_eval = 0.3
        probs = pi[actions] ** (1.0 / temperature_eval)
        probs /= np.sum(probs)
        best_action = np.random.choice(actions, p=probs)

        state, _ = game.nextState(state, best_action)
        player = game.playerId(state)

        ended, score = game.gameEnded(state)
        if ended:
            break
    else:
        score = get_dense_score(state)

    return score


def evaluate_vs_previous(net, previous_net, num_sims, c_puct, device, num_starts=20, max_steps=300):
    """
    当前网络（带搜索）与前一个模型（带搜索）对战，返回当前网络的胜率。
    平局算 0.5 胜。
    """
    np.random.seed(42)
    wins = 0.0

    for _ in range(num_starts):
        state = game.rootState()
        player = game.playerId(state)
        step = 0
        score = 0.0

        while step < max_steps:
            step += 1
            actions = game.getValidActions(state)

            net_used = net if player == 1 else previous_net

            def nnet(states):
                with torch.no_grad():
                    states_t = torch.from_numpy(states).float().to(device)
                    logits, values = net_used(states_t)
                    probs = torch.softmax(logits, dim=1)
                return probs.cpu().numpy(), values.cpu().numpy().flatten()

            root = state[np.newaxis, :]
            pi, _ = learn_pi_and_v(root, num_sims, nnet, c_puct)
            pi = pi[0]

            temperature_eval = 0.3
            probs = pi[actions] ** (1.0 / temperature_eval)
            probs /= np.sum(probs)
            best_action = np.random.choice(actions, p=probs)

            state, _ = game.nextState(state, best_action)
            player = game.playerId(state)

            ended, score = game.gameEnded(state)
            if ended:
                break
        else:
            score = get_dense_score(state)

        if score > 0:
            wins += 1.0
        elif score == 0:
            wins += 0.5

    return wins / num_starts


# ------------------------------------------------------------
# 3. 主训练循环
# ------------------------------------------------------------
def train():
    # ---- 超参数 ----
    num_sims = 1200
    c_puct = metaparm.c_puct
    batch_size = 256
    num_selfplay_games = 3
    num_epochs = 1000
    learning_rate = 0.0001
    eval_interval = 10
    num_starts = 50
    update_threshold = 0.55       # 胜率超过此值即认为有进步
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
            "update_threshold": update_threshold,
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

    # ---- 固定随机基线（仅用于参考） ----
    random_net = WatermelonGCN().to(device)
    baseline_net = copy.deepcopy(random_net)

    # ---- 🔥 历史最优（存档）和“上一个模型”（用于评估） ----
    best_net = copy.deepcopy(net)
    previous_net = copy.deepcopy(net)   # 用于评估对比的前一个模型
    selfplay_net = copy.deepcopy(best_net)

    # ---- 训练循环 ----
    for epoch in range(start_epoch, num_epochs):
        # ---- 自对弈（使用 selfplay_net 生成数据） ----
        total_steps = 0
        for idx in range(num_selfplay_games):
            def nnet(states):
                with torch.no_grad():
                    states_t = torch.from_numpy(states).float().to(device)
                    logits, values = selfplay_net(states_t)
                    probs = torch.softmax(logits, dim=1)
                return probs.cpu().numpy(), values.cpu().numpy().flatten()

            steps_in_game = 0
            for state, policy, z, ren in self_play(nnet, num_sims, c_puct):
                replay_buffer.push(state, policy, z)
                steps_in_game += 1
                total_steps += 1

            print(f"{idx} Self-play done, steps={steps_in_game}, buffer size: {len(replay_buffer)}")

        sw.log({"self_play_steps": total_steps, "epoch": epoch})
        print(f"Epoch {epoch}, total steps: {total_steps}, buffer size: {len(replay_buffer)}")

        # ---- 训练网络 ----
        if len(replay_buffer) >= batch_size:
            num_updates = min(len(replay_buffer) // batch_size, 128)

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
                pw = 1
                vw = 1
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

        # ---- 🔥 核心评估：当前网络 vs 前一个模型 ----
        if (epoch + 1) % eval_interval == 0:
            win_rate_vs_previous = evaluate_vs_previous(
                net, previous_net, num_sims, c_puct, device, num_starts=num_starts
            )
            print(f"Epoch {epoch}, Win Rate vs Previous: {win_rate_vs_previous:.3f}")
            sw.log({"win_rate_vs_previous": win_rate_vs_previous, "epoch": epoch})

            # 🔥 如果当前网络胜率 > 阈值，认为有进步：
            # 1. 更新 previous_net 为当前网络（下次评估对照新的“上一代”）
            # 2. 同步更新 best_net 和 selfplay_net（因为比上一代强，自然也比历史最优强）
            if win_rate_vs_previous > update_threshold:
                previous_net = copy.deepcopy(net)
                best_net = copy.deepcopy(net)
                selfplay_net = copy.deepcopy(best_net)
                torch.save(net.state_dict(), best_model_path)
                sw.save(best_model_path)
                print(f"🏆 模型进步！Epoch {epoch}, 胜率 {win_rate_vs_previous:.3f}")
            else:
                print(f"⏳ 未达到阈值 {update_threshold}，继续训练...")

        # ---- 定期保存模型 ----
        if epoch % 50 == 0:
            model_path = f"model_epoch_{epoch}.pth"
            torch.save(net.state_dict(), model_path)
            sw.save(model_path)

    sw.finish()
    print(f"\n🎉 训练完成！最终最佳模型保存在 {best_model_path}")


if __name__ == "__main__":
    train()