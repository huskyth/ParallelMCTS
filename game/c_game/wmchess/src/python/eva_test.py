import torch
import numpy as np
import copy
from . import game
from .RMCTS import learn_pi_and_v
from .wmnet import WatermelonNet


def play_game_with_noise(net1, net2, num_sims, c_puct, device,
                         noise_level=0.3, temperature_eval=0.5, max_steps=500):
    """
    与 play_game 相同，但 net2（对手）的采样会加入 Dirichlet 噪声。
    noise_level: 0~1，0=无噪声，1=完全随机
    """
    state = game.rootState()
    player = game.playerId(state)
    step = 0
    position_count = {}
    score = 0.0

    while step < max_steps:
        step += 1
        actions = game.getValidActions(state)
        if not actions:
            score = get_dense_score(state)
            break

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

        # ---- 对 net2 加入噪声 ----
        if player == -1 and noise_level > 0:
            # 只对后手（net2）的决策加入噪声
            noise = np.random.dirichlet([1.0] * len(actions))
            # 混合噪声：noise_level 控制噪声强度
            pi[actions] = (1 - noise_level) * pi[actions] + noise_level * noise

        # 温度采样（与 play_game 一致）
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


def test_with_noise(net, baseline_net, device, num_games=30, num_sims=200, c_puct=1.0):
    """
    对比无噪声和有噪声（对手）时的胜率。
    """
    print("=" * 60)
    print("🔍 测试：给对手加噪声能否提升胜率")
    print("=" * 60)

    # 1. 无噪声基线（原版 play_game）
    wins_no_noise = 0
    for i in range(num_games):
        result = play_game(net, baseline_net, num_sims, c_puct, device)
        if result > 0:
            wins_no_noise += 1
        elif result == 0:
            wins_no_noise += 0.5
    win_rate_no_noise = wins_no_noise / num_games
    print(f"📊 无噪声胜率: {win_rate_no_noise * 100:.1f}% ({wins_no_noise}/{num_games})")

    # 2. 给对手加噪声（net2）
    for noise_level in [0.1, 0.2, 0.3, 0.5]:
        wins_with_noise = 0
        for i in range(num_games):
            result = play_game_with_noise(net, baseline_net, num_sims, c_puct, device,
                                          noise_level=noise_level)
            if result > 0:
                wins_with_noise += 1
            elif result == 0:
                wins_with_noise += 0.5
        win_rate_with_noise = wins_with_noise / num_games
        print(f"📊 对手噪声 {noise_level * 100:.0f}% 胜率: {win_rate_with_noise * 100:.1f}% ({wins_with_noise}/{num_games})")

    print("=" * 60)
    print("💡 结论：")
    if win_rate_with_noise > win_rate_no_noise + 0.1:
        print("  ✅ 给对手加噪声后胜率显著上升 → 你的网络其实在进步，")
        print("     只是评估时两个网络都太‘死板’，容易陷入僵局。")
        print("     建议：在评估时降低温度（0.3）或增加少量随机性。")
    else:
        print("  ⚠️ 加噪声后胜率变化不大 → 网络可能确实还没学会有效策略，")
        print("     或评估机制本身已足够随机（温度已偏高）。")
    print("=" * 60)


# 使用方法（在 train.py 中或单独运行）
if __name__ == "__main__":
    # 加载你的网络（假设你有一个 saved_model.pth）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = WatermelonNet(input_dim=game.gameLength(), num_actions=game.numActions()).to(device)
    # net.load_state_dict(torch.load("model_epoch_100.pth"))  # 取消注释加载你的模型

    baseline_net = copy.deepcopy(net)  # 或者加载初始基线模型

    test_with_noise(net, baseline_net, device, num_games=30, num_sims=200, c_puct=1.0)