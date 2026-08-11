import math
import random
import numpy as np

# ---------- 游戏定义 ----------
class SimpleGame:
    """论文图2的单玩家游戏"""
    def __init__(self):
        # 我们用字符串 's' 和 't' 表示非终局状态，用 'terminal_left_s' 等表示终局
        pass

    def is_terminal(self, state):
        return state.startswith('terminal')

    def get_valid_actions(self, state):
        if state == 's':
            return ['left', 'right']
        elif state == 't':
            return ['left', 'right']
        else:
            return []

    def next_state(self, state, action):
        if state == 's':
            if action == 'left':
                return 'terminal_s_left'   # 终局值 1
            else:  # right
                return 't'
        elif state == 't':
            if action == 'left':
                return 'terminal_t_left'   # 终局值 -3
            else:  # right
                return 'terminal_t_right'  # 终局值 2
        else:
            raise ValueError("Invalid state")

    def game_score(self, state):
        """终局时的得分（相对玩家1，单玩家所以就是奖励）"""
        if state == 'terminal_s_left':
            return 1.0
        elif state == 'terminal_t_left':
            return -3.0
        elif state == 'terminal_t_right':
            return 2.0
        else:
            raise ValueError("Not a terminal state")

    def player_id(self, state):
        # 单玩家，始终+1
        return 1.0

# ---------- 算法3: 分配模拟次数 ----------
def assign_simulations(pi, N):
    """
    pi: list or np.array, 先验策略（已归一化）
    N: 总模拟次数
    返回: 每个动作分配的模拟次数（整数列表），总和为N
    """
    n = len(pi)
    pi = np.array(pi) / np.sum(pi)
    # 使用多项式分布一次生成
    counts = np.random.multinomial(N, pi)
    return counts.tolist()

# ---------- 算法4: 后验策略优化（牛顿法） ----------
def policy_optimization(Q, pi0, N, C):
    """
    Q: 列表，每个动作的估计奖励
    pi0: 列表，先验策略（已归一化）
    N: 总模拟次数（用于 sqrt(N)）
    C: 探索常数
    返回: 优化后的后验策略 pi_bar (list)
    """
    n = len(Q)
    pi0 = np.array(pi0) / np.sum(pi0)
    Q = np.array(Q)
    lam = C / math.sqrt(N) if N > 0 else 1.0  # N=0时特殊处理
    epsilon = 1e-10

    # 定义 f(u) = -1 + lam * sum(pi0[i] / (u - Q[i]))
    def f(u):
        return -1.0 + lam * np.sum(pi0 / (u - Q))

    # 导数 f'(u) = -lam * sum(pi0[i] / (u - Q[i])^2)
    def fprime(u):
        return -lam * np.sum(pi0 / ((u - Q) ** 2))

    # 初始 u = max(Q) + lam * pi0[argmax(Q)]
    u = np.max(Q) + lam * pi0[np.argmax(Q)]
    if u <= np.max(Q):
        u = np.max(Q) + 1e-6

    # 牛顿迭代
    for _ in range(100):
        val = f(u)
        if abs(val) < epsilon:
            break
        der = fprime(u)
        if der == 0:
            break
        u_new = u - val / der
        # 保证 u_new > max(Q)
        if u_new <= np.max(Q):
            u_new = np.max(Q) + 1e-6
        if abs(u_new - u) < 1e-12:
            break
        u = u_new

    # 计算后验策略
    pi_bar = lam * pi0 / (u - Q)
    pi_bar = pi_bar / np.sum(pi_bar)  # 归一化
    return pi_bar.tolist()

# ---------- 算法2: RMCTS 递归 ----------
def rmcts(state, N, C, game, prior_policy_func, prior_value_func):
    """
    state: 当前状态
    N: 分配给该状态的总模拟次数
    C: 探索常数
    game: 游戏对象
    prior_policy_func: 函数，输入state，返回先验策略 (list)
    prior_value_func: 函数，输入state，返回先验价值 (float)
    返回: (value, policy) 该状态的新价值和后验策略
    """
    # 如果终局，直接返回得分和空策略
    if game.is_terminal(state):
        return game.game_score(state), None

    # 获取先验
    pi0 = prior_policy_func(state)   # list，已归一化
    v0 = prior_value_func(state)

    # 该状态自身消耗1次模拟，剩余 N-1 次分配给子动作
    remaining = N - 1
    if remaining < 0:
        raise ValueError("N must be at least 1")

    # 分配模拟次数给子动作
    actions = game.get_valid_actions(state)
    num_actions = len(actions)
    # 只对合法动作分配
    pi0_valid = [pi0[a] for a in actions]  # 假设 actions 是整数或字符串，这里简单对应
    # 但我们需要动作标识，用索引对应
    # 我们使用动作字符串，直接构建字典
    # 先归一化
    pi0_dict = {a: pi0[a] for a in actions}
    # 分配
    counts = assign_simulations(list(pi0_dict.values()), remaining)
    # 得到每个动作的模拟次数
    sim_counts = {a: counts[i] for i, a in enumerate(actions)}

    # 递归计算每个子动作的 Q 值
    Q = {}
    for a in actions:
        if sim_counts[a] == 0:
            # 如果没有分配，则不计算，但 Q 需要定义，我们跳过，但后验策略中这些动作概率为0
            continue
        child_state = game.next_state(state, a)
        child_value, _ = rmcts(child_state, sim_counts[a], C, game, prior_policy_func, prior_value_func)
        # 注意符号：单玩家，直接取 child_value
        Q[a] = child_value

    # 现在我们有了 Q 值（仅对分配了次数的动作）
    # 构建 Q 列表和对应的动作顺序（与 pi0 对应）
    # 我们仅对分配了次数的动作计算后验策略
    actions_with_Q = [a for a in actions if a in Q]
    if not actions_with_Q:
        # 如果没有子动作，返回先验值和策略（不应该发生，因为 N>=1）
        return v0, pi0

    Q_list = [Q[a] for a in actions_with_Q]
    pi0_list = [pi0[a] for a in actions_with_Q]

    # 归一化 pi0_list（在受限动作上）
    pi0_list = np.array(pi0_list) / np.sum(pi0_list)

    # 计算后验策略（只针对这些动作）
    pi_bar_restricted = policy_optimization(Q_list, pi0_list, remaining, C)

    # 构造完整策略（所有动作）
    pi_bar = {a: 0.0 for a in actions}
    for a, p in zip(actions_with_Q, pi_bar_restricted):
        pi_bar[a] = p
    # 确保总和为1
    total = sum(pi_bar.values())
    if total > 0:
        for a in pi_bar:
            pi_bar[a] /= total

    # 计算新价值 v_bar
    v_bar = (1.0 / N) * v0 + ((N - 1) / N) * sum(Q[a] * pi_bar[a] for a in actions_with_Q)

    # 将策略转换为列表（按原始动作顺序）
    # 这里我们直接返回字典
    return v_bar, pi_bar

# ---------- 测试 ----------
def paper_example():
    random.seed(42)  # 固定随机种子以便复现
    np.random.seed(42)

    game = SimpleGame()

    # 先验策略和值（论文：均匀，v=0）
    def prior_policy(state):
        actions = game.get_valid_actions(state)
        return {a: 1.0 / len(actions) for a in actions}

    def prior_value(state):
        return 0.0  # 所有非终局先验值为0

    N_root = 1003
    C = 1.0

    v, pi = rmcts('s', N_root, C, game, prior_policy, prior_value)

    print("Root state 's'")
    print(f"Estimated value: {v:.6f}")
    print("Posterior policy:")
    for a in ['left', 'right']:
        print(f"  {a}: {pi.get(a, 0.0):.6f}")

    # 额外打印子状态 t 的策略（可选）
    # 我们递归时 t 的状态也被计算，但这里不返回，可以单独调用
    # 为了查看，我们直接计算 t
    v_t, pi_t = rmcts('t', 501, C, game, prior_policy, prior_value)
    print("\nState 't' (with 501 simulations)")
    print(f"Estimated value: {v_t:.6f}")
    for a in ['left', 'right']:
        print(f"  {a}: {pi_t.get(a, 0.0):.6f}")

if __name__ == "__main__":
    paper_example()