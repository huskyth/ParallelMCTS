from .RMCTS import learn_pi_and_v

import numpy as np
from . import game


def self_play(nnet, num_sims, c_puct, temperature=1.0, dirichlet_alpha=0.3):
    """
    使用 RMCTS 进行一盘自对弈，生成训练数据。

    参数:
        nnet: 网络推理函数，接收 (batch, state_dim) 的 numpy 数组，返回 (policies, values)
        num_sims: RMCTS 模拟次数
        c_puct: 探索常数
        temperature: 策略采样温度（1.0 为正常，0 为确定性选择）
        dirichlet_alpha: Dirichlet 噪声参数（用于增加探索）

    生成:
        (state, policy, z) 元组，其中 z 是相对于当前玩家的最终结果（+1/-1）
    """
    # 1. 初始化棋盘
    state = game.rootState()
    history = []
    player = 1  # 黑棋先走（假设 playerId 返回 +1/-1）

    while True:
        # 2. 获取合法动作
        actions = game.getValidActions(state)
        if len(actions) == 0:
            break

        # 3. RMCTS 搜索
        # 将状态转为 batch (1, gameLength)
        root = state[np.newaxis, :]
        pi, _ = learn_pi_and_v(root, num_sims, nnet, c_puct)
        pi = pi[0]  # 去掉 batch 维度

        # 4. 添加 Dirichlet 噪声（仅对合法动作）
        if temperature == 1.0:
            noise = np.random.dirichlet([dirichlet_alpha] * len(actions))
            for i, a in enumerate(actions):
                pi[a] = 0.75 * pi[a] + 0.25 * noise[i]

        # 5. 采样动作（或按温度选择）
        if temperature == 0:
            a = actions[np.argmax(pi[actions])]
        else:
            probs = pi[actions] ** (1.0 / temperature)
            probs /= np.sum(probs)
            a = np.random.choice(actions, p=probs)

        # 6. 记录历史 (state, policy, player)
        history.append((state.copy(), pi.copy(), player))

        # 7. 执行动作
        state = game.nextState(state, a)
        # 切换玩家（注意：nextState 可能已经切换了玩家，我们根据 playerId 更新）
        player = game.playerId(state)  # 获取当前轮到谁（+1 或 -1）

        # 8. 检查终局
        ended, score = game.gameEnded(state)
        if ended:
            break

    # 9. 计算每个历史步骤的 z（相对于当时走棋的玩家）
    z_abs = score   # 终局得分（相对玩家1）
    for s, p, pl in history:
        z = z_abs * pl   # pl 是当时走棋的玩家（+1/-1）
        yield s, p, z