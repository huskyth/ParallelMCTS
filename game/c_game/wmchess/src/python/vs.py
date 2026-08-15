# test_best_vs_random.py
import torch
import numpy as np
from . import game
from .wmnet import WatermelonNet
from .RMCTS import learn_pi_and_v

def get_dense_score(state):
    board = state[1:]
    black = sum(1 for x in board if x == 1)
    white = sum(1 for x in board if x == -1)
    return (black - white) / 21.0

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

        temperature_eval = 0.05
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


def evaluate_best_vs_random(best_model_path, num_games=100, num_sims=200, c_puct=1.0, device=None):
    """
    加载最优模型，与随机初始化网络对弈。
    返回胜率（最优模型为先手）。
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. 加载最优模型
    best_net = WatermelonNet(input_dim=game.gameLength(), num_actions=game.numActions()).to(device)
    best_net.load_state_dict(torch.load(best_model_path, map_location=device))
    best_net.eval()

    # 2. 创建随机网络（权重随机）
    random_net = WatermelonNet(input_dim=game.gameLength(), num_actions=game.numActions()).to(device)
    random_net.eval()

    # 3. 对弈
    wins = 0
    for i in range(num_games):
        result = play_game(best_net, random_net, num_sims, c_puct, device)
        if result > 0:
            wins += 1
        elif result == 0:
            wins += 0.5  # 平局算半胜

    win_rate = wins / num_games
    print(f"🏆 最优模型 vs 随机网络 胜率: {win_rate * 100:.1f}% ({wins}/{num_games})")
    return win_rate


def evaluate_random_vs_random(num_games=100, num_sims=200, c_puct=1.0, device=None):
    """
    基线测试：两个随机网络对弈，胜率应接近 50%。
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    net1 = WatermelonNet(input_dim=game.gameLength(), num_actions=game.numActions()).to(device)
    net2 = WatermelonNet(input_dim=game.gameLength(), num_actions=game.numActions()).to(device)
    net1.eval()
    net2.eval()

    wins = 0
    for i in range(num_games):
        result = play_game(net1, net2, num_sims, c_puct, device)
        if result > 0:
            wins += 1
        elif result == 0:
            wins += 0.5

    win_rate = wins / num_games
    print(f"🎲 随机 vs 随机 胜率: {win_rate * 100:.1f}% ({wins}/{num_games})")
    return win_rate


if __name__ == "__main__":
    # 注意：请确保 best_model.pth 在当前目录或提供完整路径
    evaluate_best_vs_random("best_model.pth", num_games=100, num_sims=200, c_puct=1.0)
    # evaluate_random_vs_random(num_games=100, num_sims=200, c_puct=1.0)
