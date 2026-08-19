# evaluate_model.py
import sys
import os
import torch
import numpy as np
import json
import ctypes
from argparse import ArgumentParser

# 确保能导入项目模块（假设 build/wmchess 在 sys.path）
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'build', 'wmchess'))

from . import game
from .RMCTS import learn_pi_and_v
from .wmnet_gcn import WatermelonGCN
from .astar_player import computerMove

# ---------- 辅助函数 ----------
def get_dense_score(state):
    board = state[1:]
    black = sum(1 for x in board if x == 1)
    white = sum(1 for x in board if x == -1)
    return 3.0 * (black - white) / 21.0   # 注意：与训练时一致，已放大3倍

def get_action_index_from_move(state, from_idx, to_idx):
    actions = game.getValidActions(state)
    player = state[0]
    for a in actions:
        test_state = np.copy(state)
        test_state, _ = game.nextState(test_state, a)
        if test_state[from_idx + 1] == 0 and test_state[to_idx + 1] == player:
            return a
    return -1

# ---------- 对战函数 ----------
def play_vs_random(net, num_sims, c_puct, device, max_steps=300):
    """
    一局：当前网络（带搜索）执黑，纯随机对手（无搜索）执白。
    返回得分（>0 表示网络赢）。
    """
    state = game.rootState()
    player = game.playerId(state)
    step = 0
    score = 0.0
    while step < max_steps:
        step += 1
        actions = game.getValidActions(state)
        if player == 1:
            # 网络走棋
            def nnet(states):
                with torch.no_grad():
                    states_t = torch.from_numpy(states).float().to(device)
                    logits, values = net(states_t)
                    probs = torch.softmax(logits, dim=1)
                return probs.cpu().numpy(), values.cpu().numpy().flatten()
            root = state[np.newaxis, :]
            pi, _ = learn_pi_and_v(root, num_sims, nnet, c_puct)
            pi = pi[0]
            # 温度 0.3 采样
            temperature_eval = 0.3
            probs = pi[actions] ** (1.0 / temperature_eval)
            probs /= np.sum(probs)
            best_action = np.random.choice(actions, p=probs)
        else:
            # 随机走
            best_action = np.random.choice(actions)
        state, _ = game.nextState(state, best_action)
        player = game.playerId(state)
        ended, score = game.gameEnded(state)
        if ended:
            break
    else:
        score = get_dense_score(state)
    return score

def play_vs_astar(net, num_sims, c_puct, device, max_steps=300):
    """
    一局：当前网络（带搜索）执黑，AStar 启发式 AI 执白。
    """
    state = game.rootState()
    player = game.playerId(state)
    step = 0
    score = 0.0
    while step < max_steps:
        step += 1
        actions = game.getValidActions(state)
        if player == 1:
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

        else:
            point_status = state[1:].tolist()
            best_move, _ = computerMove(point_status, 2)
            if best_move is None:
                print(f'None state = {state}')
            from_idx, to_idx = best_move

            action = get_action_index_from_move(state, from_idx, to_idx)
            if action == -1:
                break
            best_action = action
        state, _ = game.nextState(state, best_action)
        player = game.playerId(state)
        ended, score = game.gameEnded(state)
        if ended:
            print(f"结束 {score}, state = {state}")
            break
    else:
        score = get_dense_score(state)
        print(f"和棋, score = {score}")

    return score

# ---------- 批量评估函数 ----------
def evaluate_against(net, opponent_type, num_games, num_sims, c_puct, device):
    """
    opponent_type: 'random' 或 'astar'
    返回 (胜率, 平均得分)
    """
    wins = 0.0
    total_score = 0.0
    for i in range(num_games):
        if opponent_type == 'random':
            score = play_vs_random(net, num_sims, c_puct, device)
        else:
            score = play_vs_astar(net, num_sims, c_puct, device)
        total_score += score
        if score > 0:
            wins += 1.0
        elif score == 0:
            wins += 0.5
        if (i+1) % 20 == 0:
            print(f"  已完成 {i+1}/{num_games} 局，当前胜率: {wins/(i+1):.3f}")
    win_rate = wins / num_games
    avg_score = total_score / num_games
    return win_rate, avg_score

# ---------- 主函数 ----------
def main():
    parser = ArgumentParser(description="评估西瓜棋模型")
    parser.add_argument('--model', type=str, default='best_model.pth', help='模型文件路径')
    parser.add_argument('--opponent', type=str, choices=['random', 'astar'], default='astar', help='对手类型')
    parser.add_argument('--games', type=int, default=100, help='对局数')
    parser.add_argument('--sims', type=int, default=1200, help='搜索次数')
    parser.add_argument('--c_puct', type=float, default=1.0, help='探索常数')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    device = torch.device(args.device)

    # 加载网络
    net = WatermelonGCN().to(device)
    if not os.path.exists(args.model):
        print(f"❌ 模型文件 {args.model} 不存在！")
        sys.exit(1)
    net.load_state_dict(torch.load(args.model, map_location=device))
    net.eval()
    print(f"✅ 加载模型: {args.model}")

    # 评估
    print(f"🔄 评估中，对手: {args.opponent}, 对局数: {args.games}, 搜索次数: {args.sims}")
    win_rate, avg_score = evaluate_against(
        net, args.opponent, args.games, args.sims, args.c_puct, device
    )
    print(f"\n📊 评估结果:")
    print(f"  胜率: {win_rate*100:.1f}%")
    print(f"  平均得分: {avg_score:.4f}")

    # 可选：保存 ELO（如果之前有 ELO 文件）
    # 简单计算 ELO（假设对手 ELO 为 1200）
    if args.opponent == 'astar':
        opponent_elo = 1200  # 可调
        current_elo = 1200
        expected = 1.0 / (1.0 + 10 ** ((opponent_elo - current_elo) / 400.0))
        new_elo = current_elo + 32 * (win_rate - expected)
        print(f"  估算 ELO (初始 1200): {new_elo:.0f}")

if __name__ == '__main__':
    main()