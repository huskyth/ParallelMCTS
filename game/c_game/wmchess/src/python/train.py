# train.py
import numpy as np
import torch
import torch.optim as optim
import swanlab as sw
import copy
import random
import os
from . import game
from .RMCTS import learn_pi_and_v
from .wmnet import WatermelonNet
from .replay_buf import ReplayBuffer
from . import metaparm

sw.login(api_key="rdGaOSnlBY0KBDnNdkzja")


# ------------------------------------------------------------
# 1. 自对弈函数
# ------------------------------------------------------------
def self_play(nnet, num_sims, c_puct, temperature=1.0, dirichlet_alpha=0.3, max_steps=500):
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
        state = game.nextState(state, a)
        player = game.playerId(state)

        ended, score = game.gameEnded(state)
        if ended:
            reason = 'e'
            break

        s = ','.join(str(int(x)) for x in state.tolist())
        position_count[s] = position_count.get(s, 0) + 1
        if position_count[s] >= 3:
            score = get_dense_score(state)
            score = max(-1.0, min(1.0, score))
            reason = 'r'
            break
    else:
        score = get_dense_score(state)
        reason = 't'

    if abs(score) < 1e-6:
        return

    z_abs = score
    for s, p, pl in history:
        z = z_abs * pl
        yield s, p, z, reason


def get_dense_score(state):
    board = state[1:]
    black = sum(1 for x in board if x == 1)
    white = sum(1 for x in board if x == -1)
    return (black - white) / 21.0


# ------------------------------------------------------------
# 2. 对战与评估函数
# ------------------------------------------------------------
def play_game(net1, net2, num_sims, c_puct, device, max_steps=500):
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
            break

        s = ','.join(str(int(x)) for x in state.tolist())
        position_count[s] = position_count.get(s, 0) + 1
        if position_count[s] >= 3:
            score = get_dense_score(state)
            break
    else:
        score = get_dense_score(state)

    return score


def evaluate(net, baseline_net, num_games, num_sims, c_puct, device, max_steps=500):
    wins = 0
    for i in range(num_games):
        result = play_game(net, baseline_net, num_sims, c_puct, device, max_steps=max_steps)
        if result > 0:
            wins += 1
        elif result == 0:
            wins += 0.5
    return wins / num_games


# ------------------------------------------------------------
# 3. 主训练循环
# ------------------------------------------------------------
def train():
    # ---- 超参数 ----
    num_sims = 200
    c_puct = metaparm.c_puct
    batch_size = 256
    num_selfplay_games = 32
    num_epochs = 1000
    learning_rate = 0.0001
    eval_interval = 25
    eval_games = 100
    eval_sims = 200
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    best_model_path = "best_model.pth"

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

    # 🔥 从 best_model.pth 恢复继续训练
    start_epoch = 0
    if os.path.exists(best_model_path):
        print(f"📂 发现已有 best_model.pth，加载权重继续训练...")
        net.load_state_dict(torch.load(best_model_path, map_location=device))
    else:
        print("🆕 未找到 best_model.pth，从随机初始化开始训练。")

    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    replay_buffer = ReplayBuffer(max_size=200000)

    # ---- 固定随机基线（用于参考评估） ----
    random_net = WatermelonNet(input_dim=game.gameLength(), num_actions=game.numActions()).to(device)
    baseline_net = copy.deepcopy(random_net)  # 固定，不更新

    # ---- 历史最优网络（用于评估和生成数据） ----
    best_net = copy.deepcopy(net)          # 初始为当前网络
    selfplay_net = copy.deepcopy(best_net) # 用于生成自对弈数据，与 best_net 同步

    # ---- 训练循环 ----
    for epoch in range(start_epoch, num_epochs):
        # ---- 自对弈（使用 selfplay_net 生成数据） ----
        rep = 0
        for idx in range(num_selfplay_games):
            def nnet(states):
                with torch.no_grad():
                    states_t = torch.from_numpy(states).float().to(device)
                    logits, values = selfplay_net(states_t)   # 🔥 使用历史最优模型生成数据
                    probs = torch.softmax(logits, dim=1)
                return probs.cpu().numpy(), values.cpu().numpy().flatten()

            for state, policy, z, ren in self_play(nnet, num_sims, c_puct):
                replay_buffer.push(state, policy, z)
                if ren == 'r':
                    rep += 1

            print(f"{idx} Self-play done, buffer size: {len(replay_buffer)}")

        print(f"Epoch {epoch}, buffer size: {len(replay_buffer)}")

        # ---- 训练网络（当前 net 拟合自对弈数据） ----
        if len(replay_buffer) >= batch_size:
            if len(replay_buffer) < 50000:
                num_updates = min(len(replay_buffer) // batch_size, 1)
            else:
                num_updates = min(len(replay_buffer) // batch_size, 1)

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
                beta = 0.06
                entropy_loss = -beta * entropy
                loss = policy_loss + value_loss + entropy_loss

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

        # ---- 定期评估 ----
        if (epoch + 1) % eval_interval == 0:
            # 1. vs 随机基线（参考）
            win_rate_vs_random = evaluate(net, baseline_net, eval_games, eval_sims, c_puct, device)
            print(f"Epoch {epoch}, Win Rate vs Random: {win_rate_vs_random:.3f}")
            sw.log({"win_rate_vs_random": win_rate_vs_random, "epoch": epoch})

            # 2. 🔥 当前网络 vs 历史最优
            win_rate_vs_best = evaluate(net, best_net, eval_games, eval_sims, c_puct, device)
            print(f"Epoch {epoch}, Win Rate vs Best: {win_rate_vs_best:.3f}")
            sw.log({"win_rate_vs_best": win_rate_vs_best, "epoch": epoch})

            # 3. 如果当前网络击败了历史最优（胜率 > 0.55），更新 best_net 和 selfplay_net
            if win_rate_vs_best > 0.55:
                best_net = copy.deepcopy(net)
                selfplay_net = copy.deepcopy(best_net)   # 🔥 同步自对弈生成器
                torch.save(net.state_dict(), best_model_path)
                sw.save(best_model_path)
                print(f"🏆 更新历史最优模型！Epoch {epoch}, 胜率 {win_rate_vs_best:.3f}")

        # ---- 定期保存（每50轮） ----
        if epoch % 50 == 0:
            model_path = f"model_epoch_{epoch}.pth"
            torch.save(net.state_dict(), model_path)
            sw.save(model_path)

    sw.finish()
    print(f"\n🎉 训练完成！最终最佳模型保存在 {best_model_path}")


if __name__ == "__main__":
    train()