# test_random_vs_pure_random.py
import numpy as np
import torch
from . import game
from .RMCTS import learn_pi_and_v
from .wmnet import WatermelonNet
from . import metaparm


# 复制 get_dense_score 定义
def get_dense_score(state):
    board = state[1:]
    black = sum(1 for x in board if x == 1)
    white = sum(1 for x in board if x == -1)
    return (black - white) / 21.0


# 复制 evaluate_vs_pure_random 函数（或从 train 导入，但为了避免循环依赖，这里直接复制）
def evaluate_vs_pure_random(net, num_sims, c_puct, device, num_starts=5, max_steps=300):
    np.random.seed(42)
    scores = []
    for _ in range(num_starts):
        state = game.rootState()
        steps = np.random.randint(5, 15)
        for _ in range(steps):
            actions = game.getValidActions(state)
            if not actions:
                break
            a = np.random.choice(actions)
            state = game.nextState(state, a)
            ended, _ = game.gameEnded(state)
            if ended:
                break
        player = game.playerId(state)
        step = 0
        score = 0.0
        while step < max_steps:
            step += 1
            actions = game.getValidActions(state)
            if not actions:
                score = -1.0 if player == 1 else 1.0
                break
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
                best_action = actions[np.argmax(pi[actions])]
            else:
                best_action = np.random.choice(actions)
            state = game.nextState(state, best_action)
            player = game.playerId(state)
            ended, score = game.gameEnded(state)
            if ended:
                break
        else:
            score = get_dense_score(state)
        scores.append(score)
    return np.mean(scores)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 创建随机网络
    net = WatermelonNet(input_dim=game.gameLength(), num_actions=game.numActions()).to(device)
    # 设置评估参数
    num_sims = 400  # 与训练一致
    c_puct = metaparm.c_puct  # 假设 metaparm 中有定义
    num_starts = 5
    max_steps = 300
    avg_score = evaluate_vs_pure_random(net, num_sims, c_puct, device, num_starts, max_steps)
    print(f"随机网络 vs 纯随机（无搜索）平均得分: {avg_score:.4f}")
    print("如果得分接近 0，说明随机网络并不比纯随机强，这是预期的。")


if __name__ == "__main__":
    main()
