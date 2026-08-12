from .RMCTS import learn_pi_and_v
import numpy as np

def self_play(game, net, num_sims, c_puct, temperature=1.0, dirichlet_alpha=0.3):
    state = np.zeros(game.gameLength(), dtype=np.float32)
    # 初始化棋盘 (黑棋在 0,1,2,3,4,8; 白棋 7,11,12,13,14,15)
    state[0] = 1.0  # 当前玩家黑
    black_init = [0,1,2,3,4,8]
    white_init = [7,11,12,13,14,15]
    for i in black_init: state[i+1] = 1.0
    for i in white_init: state[i+1] = -1.0

    history = []
    while True:
        actions = np.zeros(game.numActions(), dtype=np.int32)
        num_legal = game.getValidActions(actions, state)
        if num_legal == 0:
            break

        # 获取搜索策略
        # 注意 learn_pi_and_v 需要 G 是二维数组 (batch, gamesize)
        G = state[np.newaxis, :]
        pi, _ = learn_pi_and_v(G, num_sims, net, c_puct)
        pi = pi[0]  # 去掉 batch

        # 添加噪声
        if temperature == 1.0:
            noise = np.random.dirichlet([dirichlet_alpha] * num_legal)
            for i, act in enumerate(actions[:num_legal]):
                pi[act] = 0.75 * pi[act] + 0.25 * noise[i]

        # 采样动作
        probs = pi[actions[:num_legal]]
        probs = np.power(probs, 1.0/temperature)
        probs /= np.sum(probs)
        action = np.random.choice(actions[:num_legal], p=probs)

        history.append((state.copy(), pi.copy(), state[0]))  # state[0] 是当前玩家

        new_state = np.zeros(game.gameLength(), dtype=np.float32)
        ret = game.nextState(new_state, state, action)
        state = new_state

        score = np.float32(0.0)
        ended = game.gameEnded(score, state)
        if ended:
            break

    z_abs = score
    for s, p, pl in history:
        z = z_abs * pl
        yield s, p, z