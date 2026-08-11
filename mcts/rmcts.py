import numpy as np
import random
from typing import List, Tuple, Any
import torch.nn as nn
from torch.optim import Adam
import torch

class RMCTS:
    """
    纯 Python 实现的 RMCTS（无 C 依赖）
    用法：传入一个实现了 game 接口的对象和神经网络推理函数
    """

    def __init__(self, game, num_sims: int, c_puct: float, neural_net):
        """
        game: 必须实现 next_state, get_valid_actions, game_ended, player_id, num_actions
        neural_net: 函数，输入状态列表，返回 (policies, values)
        """
        self.game = game
        self.num_sims = num_sims
        self.c_puct = c_puct
        self.neural_net = neural_net  # 你的 PyTorch 模型推理函数

        self.num_actions = game.num_actions()
        self.gamesize = game.game_length()

    # ---------- 核心数学：牛顿法求后验策略（完全照搬 C 代码逻辑）----------
    def _compute_posterior_policy(self, pi0: np.ndarray, Q: np.ndarray, T: int) -> np.ndarray:
        """对应 C 的 new_policy_common_ucb_Newton"""
        n = len(pi0)
        pi0 = pi0 / np.sum(pi0)
        c0 = self.c_puct / np.sqrt(max(T, 1))
        Q_max = np.max(Q)

        # 初始 delta
        delta = c0 * pi0[np.argmax(Q)]
        if delta < 1e-12:
            delta = 1e-12

        epsilon = 1e-12
        for _ in range(100):  # 牛顿迭代
            denom = (Q_max - Q) + delta
            x = (c0 * pi0) / denom
            f = np.sum(x) - 1.0
            if f <= 0:
                break
            f_prime = -np.sum(x / denom)  # 导数
            new_delta = delta - f / f_prime
            if new_delta <= delta:
                break
            delta = new_delta

        pi1 = (c0 * pi0) / ((Q_max - Q) + delta)
        return pi1 / np.sum(pi1)

    # ---------- 批量分配模拟次数（对应 C 的 assign_simulations）----------
    def _assign_simulations(self, pi: np.ndarray, budget: int) -> np.ndarray:
        """用多项式分布一次性分配，保证总数=budget"""
        pi = pi / np.sum(pi)
        return np.random.multinomial(budget, pi)

    # ---------- 主搜索入口（对应 C 的 flush + propagate）----------
    def search(self, root_states: List) -> Tuple[List[np.ndarray], List[float]]:
        """
        输入：一批根状态（list）
        输出：每个根状态的 (后验策略 new_policy, 新价值 new_value)
        """
        num_lanes = len(root_states)
        # ---------- 数据结构（模拟 C 的内存池）----------
        G = []  # 状态池
        policy = []  # 先验策略池
        value = []  # 价值池
        Q = []  # 动作 Q 值池 (num_nodes x num_actions)
        N = []  # 动作访问次池
        parent = []  # 父节点索引
        a0 = []  # 从父节点到本节点的动作
        sims = []  # 每个节点的总模拟预算
        sims_remaining = []  # 剩余模拟数

        # 初始化根节点
        for state in root_states:
            idx = len(G)
            G.append(state)
            # 根节点先调用神经网络获得先验
            p, v = self.neural_net([state])  # 假设返回 (batch_policy, batch_value)
            policy.append(p[0])
            value.append(v[0])
            Q.append(np.zeros(self.num_actions, dtype=np.float32))
            N.append(np.zeros(self.num_actions, dtype=np.int32))
            parent.append(-1)
            a0.append(-1)
            sims.append(self.num_sims)
            sims_remaining.append(self.num_sims)

        # ---------- 阶段1：批量建树（对应 MCTS_flush_new_stack）----------
        new_stack = list(range(num_lanes))  # 待扩展栈

        while new_stack:
            idx = new_stack.pop()
            state = G[idx]
            total_sims = sims[idx]

            # 叶子节点（预算只有1）直接跳过
            if total_sims == 1:
                continue

            # 合法化先验策略
            valid_actions = self.game.get_valid_actions(state)
            pi_legal = np.zeros(self.num_actions)
            pi_legal[valid_actions] = policy[idx]
            pi_legal = pi_legal / np.sum(pi_legal)

            # 分配子节点预算（总预算 - 1 分给子节点）
            action_counts = self._assign_simulations(pi_legal, total_sims - 1)

            # 生成子节点
            for a in range(self.num_actions):
                count = action_counts[a]
                if count == 0:
                    continue

                # 生成子状态
                child_state = self.game.next_state(state, a)
                child_idx = len(G)

                # 存入状态池
                G.append(child_state)
                parent.append(idx)
                a0.append(a)
                sims.append(count)
                sims_remaining.append(count)

                # 检查是否终局
                ended, score = self.game.game_ended(child_state)
                if ended:
                    # 终局节点：直接赋值，不进 inference_stack
                    player = self.game.player_id(child_state)
                    value.append(score * player)
                    policy.append(np.zeros(self.num_actions))  # 占位
                    Q.append(np.zeros(self.num_actions))
                    N.append(np.zeros(self.num_actions))
                else:
                    # 非终局：先占位，稍后统一推理
                    value.append(0.0)
                    policy.append(np.zeros(self.num_actions))  # 稍后填充
                    Q.append(np.zeros(self.num_actions))
                    N.append(np.zeros(self.num_actions))
                    # 压入待推理栈（对应 inference_stack）
                    # 但为了简化，我们这里先收集所有叶子，后面统一批量推理
                    # 注意：C 代码中推理栈用于逐层扩展，这里我们改用分批收集
                    # 为了严格对应，我们先把非终局子节点加入栈继续扩展
                    new_stack.append(child_idx)

        # ---------- 批量推理所有叶子节点的网络值 ----------
        # 找出所有 sims == 1 的非终局节点（叶子）
        leaf_indices = [i for i in range(len(G)) if sims[i] == 1 and not self.game.game_ended(G[i])[0]]
        if leaf_indices:
            leaf_states = [G[i] for i in leaf_indices]
            leaf_p, leaf_v = self.neural_net(leaf_states)
            for idx, p, v in zip(leaf_indices, leaf_p, leaf_v):
                # 合法化先验
                valid = self.game.get_valid_actions(G[idx])
                p_legal = np.zeros(self.num_actions)
                p_legal[valid] = p
                p_legal = p_legal / np.sum(p_legal)
                policy[idx] = p_legal
                value[idx] = v

        # ---------- 阶段2：自底向上传播（对应 MCTS_propagate_all）----------
        for idx in range(len(G) - 1, num_lanes - 1, -1):
            if sims[idx] == 1:
                # 叶子节点：直接用网络价值回传
                v_i = value[idx]
                sims_i = 1
            else:
                # 内部节点：用牛顿法计算新价值
                # 注意：此时子节点已全部回传，Q/N 已更新
                # 提取被访问过的动作
                N_i = N[idx]
                visited_actions = [a for a in range(self.num_actions) if N_i[a] > 0]
                if not visited_actions:
                    # 极端情况：没有子节点回传（理论上不会发生）
                    v_i = value[idx]
                    sims_i = 1
                else:
                    pi0_masked = policy[idx][visited_actions]
                    Q_masked = Q[idx][visited_actions]
                    pi0_masked = pi0_masked / np.sum(pi0_masked)
                    # 牛顿法求后验
                    pi1_masked = self._compute_posterior_policy(pi0_masked, Q_masked, sims[idx] - 1)
                    # 计算新价值
                    v_i = np.sum(pi1_masked * Q_masked)
                    # 混合网络先验（对应 C 的 v += (v0-v)/(T+1)）
                    v0 = value[idx]
                    v_i = v_i + (v0 - v_i) / sims[idx]
                    sims_i = sims[idx]

            # 回传给父节点（对应 update_parent）
            par = parent[idx]
            action = a0[idx]
            if par >= 0:
                # 更新父节点的 Q 和 N
                player_par = self.game.player_id(G[par])
                player_child = self.game.player_id(G[idx])
                v_scaled = v_i * player_child * player_par
                old_Q = Q[par][action]
                old_N = N[par][action]
                # 增量更新
                Q[par][action] = (old_Q * old_N + v_scaled * sims_i) / (old_N + sims_i)
                N[par][action] += sims_i
                sims_remaining[par] -= sims_i

        # ---------- 最终处理根节点 ----------
        new_policies = []
        new_values = []
        for idx in range(num_lanes):
            # 根节点计算后验策略和价值
            N_i = N[idx]
            visited_actions = [a for a in range(self.num_actions) if N_i[a] > 0]
            pi0_masked = policy[idx][visited_actions]
            Q_masked = Q[idx][visited_actions]
            pi0_masked = pi0_masked / np.sum(pi0_masked)
            pi1_masked = self._compute_posterior_policy(pi0_masked, Q_masked, sims[idx] - 1)

            # 还原完整策略
            pi_final = np.zeros(self.num_actions)
            for a, p in zip(visited_actions, pi1_masked):
                pi_final[a] = p

            # 计算根节点价值
            v_final = np.sum(pi1_masked * Q_masked)
            v0 = value[idx]
            v_final = v_final + (v0 - v_final) / sims[idx]

            new_policies.append(pi_final)
            new_values.append(v_final)

        return new_policies, new_values



import numpy as np
import random
from typing import List, Tuple

# ---------- 游戏定义（论文图2） ----------
class SimpleGame:
    """单玩家游戏：s -> left 终局值1，s -> right -> t；t -> left 终局值-3，t -> right 终局值2"""
    def num_actions(self):
        return 2  # left, right

    def game_length(self):
        return 1  # 无关紧要

    def get_valid_actions(self, state):
        if state == 's' or state == 't':
            return [0, 1]  # 0:left, 1:right
        else:
            return []

    def next_state(self, state, action):
        if state == 's':
            if action == 0:      # left
                return 'terminal_s_left'
            else:                # right
                return 't'
        elif state == 't':
            if action == 0:      # left
                return 'terminal_t_left'
            else:                # right
                return 'terminal_t_right'
        else:
            return state

    def game_ended(self, state):
        if state.startswith('terminal'):
            return True, self.game_score(state)
        else:
            return False, 0.0

    def game_score(self, state):
        if state == 'terminal_s_left':
            return 1.0
        elif state == 'terminal_t_left':
            return -3.0
        elif state == 'terminal_t_right':
            return 2.0
        else:
            return 0.0

    def player_id(self, state):
        return 1.0  # 单玩家，始终为正


# ---------- 模拟神经网络（返回均匀策略和零价值） ----------
def mock_neural_net(states):
    """
    输入：状态列表
    输出：(policies, values) 两个列表，长度与输入相同
    """
    batch_size = len(states)
    policies = []
    values = []
    for state in states:
        if state == 's' or state == 't':
            # 均匀策略
            policies.append(np.array([0.5, 0.5], dtype=np.float32))
            values.append(0.0)
        else:
            # 终局状态理论上不会送进网络，但以防万一
            policies.append(np.array([0.0, 0.0], dtype=np.float32))
            values.append(0.0)
    return policies, values


# ---------- 对 RMCTS 类打补丁（临时修复策略填充问题） ----------
# 为了演示，我们直接修改 search 方法内的 while 循环
# 在扩展节点前，如果 policy[idx] 全零，则调用网络获取
def patched_search(self, root_states):
    """
    与原始 search 逻辑相同，但在扩展节点前强制确保策略已填充。
    这里我们不重写整个类，而是定义一个包装函数。
    """
    # 复制原始 search 代码，并加入补丁（见下方注释）
    # 为了简便，我们直接继承类并重写 search，或使用猴子补丁。
    # 这里我们选择：在调用 search 前，动态替换方法（仅演示）
    pass


# 但我们为了快速演示，直接修改 RMCTS 类的 search 方法（运行时替换）
def search_with_fix(self, root_states):
    num_lanes = len(root_states)
    G = []
    policy = []
    value = []
    Q = []
    N = []
    parent = []
    a0 = []
    sims = []
    sims_remaining = []

    # 初始化根节点
    for state in root_states:
        idx = len(G)
        G.append(state)
        p, v = self.neural_net([state])
        policy.append(p[0])
        value.append(v[0])
        Q.append(np.zeros(self.num_actions, dtype=np.float32))
        N.append(np.zeros(self.num_actions, dtype=np.int32))
        parent.append(-1)
        a0.append(-1)
        sims.append(self.num_sims)
        sims_remaining.append(self.num_sims)

    new_stack = list(range(num_lanes))

    while new_stack:
        idx = new_stack.pop()
        state = G[idx]
        total_sims = sims[idx]

        if total_sims == 1:
            continue

        # ---------- 补丁：如果策略全零，则调用网络 ----------
        if np.all(policy[idx] == 0):
            p, v = self.neural_net([state])
            policy[idx] = p[0]
            value[idx] = v[0]

        valid_actions = self.game.get_valid_actions(state)
        pi_legal = np.zeros(self.num_actions)
        pi_legal[valid_actions] = policy[idx]
        pi_legal = pi_legal / np.sum(pi_legal)

        action_counts = self._assign_simulations(pi_legal, total_sims - 1)

        for a in range(self.num_actions):
            count = action_counts[a]
            if count == 0:
                continue

            child_state = self.game.next_state(state, a)
            child_idx = len(G)

            G.append(child_state)
            parent.append(idx)
            a0.append(a)
            sims.append(count)
            sims_remaining.append(count)

            ended, score = self.game.game_ended(child_state)
            if ended:
                player = self.game.player_id(child_state)
                value.append(score * player)
                policy.append(np.zeros(self.num_actions))
                Q.append(np.zeros(self.num_actions))
                N.append(np.zeros(self.num_actions))
            else:
                # 非终局，先占位，稍后统一推理（但此处我们会在下次扩展前补丁填充）
                value.append(0.0)
                policy.append(np.zeros(self.num_actions))
                Q.append(np.zeros(self.num_actions))
                N.append(np.zeros(self.num_actions))
                new_stack.append(child_idx)

    # ---------- 批量推理所有叶子节点（sims==1的非终局） ----------
    leaf_indices = [i for i in range(len(G)) if sims[i] == 1 and not self.game.game_ended(G[i])[0]]
    if leaf_indices:
        leaf_states = [G[i] for i in leaf_indices]
        leaf_p, leaf_v = self.neural_net(leaf_states)
        for idx, p, v in zip(leaf_indices, leaf_p, leaf_v):
            valid = self.game.get_valid_actions(G[idx])
            p_legal = np.zeros(self.num_actions)
            p_legal[valid] = p
            p_legal = p_legal / np.sum(p_legal)
            policy[idx] = p_legal
            value[idx] = v

    # ---------- 自底向上传播 ----------
    for idx in range(len(G) - 1, num_lanes - 1, -1):
        if sims[idx] == 1:
            v_i = value[idx]
            sims_i = 1
        else:
            N_i = N[idx]
            visited_actions = [a for a in range(self.num_actions) if N_i[a] > 0]
            if not visited_actions:
                v_i = value[idx]
                sims_i = 1
            else:
                pi0_masked = policy[idx][visited_actions]
                Q_masked = Q[idx][visited_actions]
                pi0_masked = pi0_masked / np.sum(pi0_masked)
                pi1_masked = self._compute_posterior_policy(pi0_masked, Q_masked, sims[idx] - 1)
                v_i = np.sum(pi1_masked * Q_masked)
                v0 = value[idx]
                v_i = v_i + (v0 - v_i) / sims[idx]
                sims_i = sims[idx]

        par = parent[idx]
        action = a0[idx]
        if par >= 0:
            player_par = self.game.player_id(G[par])
            player_child = self.game.player_id(G[idx])
            v_scaled = v_i * player_child * player_par
            old_Q = Q[par][action]
            old_N = N[par][action]
            Q[par][action] = (old_Q * old_N + v_scaled * sims_i) / (old_N + sims_i)
            N[par][action] += sims_i
            sims_remaining[par] -= sims_i

    # ---------- 根节点后验 ----------
    new_policies = []
    new_values = []
    for idx in range(num_lanes):
        N_i = N[idx]
        visited_actions = [a for a in range(self.num_actions) if N_i[a] > 0]
        pi0_masked = policy[idx][visited_actions]
        Q_masked = Q[idx][visited_actions]
        pi0_masked = pi0_masked / np.sum(pi0_masked)
        pi1_masked = self._compute_posterior_policy(pi0_masked, Q_masked, sims[idx] - 1)

        pi_final = np.zeros(self.num_actions)
        for a, p in zip(visited_actions, pi1_masked):
            pi_final[a] = p

        v_final = np.sum(pi1_masked * Q_masked)
        v0 = value[idx]
        v_final = v_final + (v0 - v_final) / sims[idx]

        new_policies.append(pi_final)
        new_values.append(v_final)

    return new_policies, new_values


# 将补丁方法绑定到 RMCTS 类
RMCTS.search = search_with_fix


# ---------- 主程序 ----------
def main():
    random.seed(42)
    np.random.seed(42)

    game = SimpleGame()
    num_sims = 1003
    c_puct = 1.0

    # 模拟神经网络
    def neural_net(states):
        return mock_neural_net(states)

    # 创建 RMCTS 实例
    rmcts = RMCTS(game, num_sims, c_puct, neural_net)

    # 只搜索根状态 's'
    root_states = ['s']
    new_policies, new_values = rmcts.search(root_states)

    print("Root state 's':")
    print(f"  Estimated value: {new_values[0]:.6f}")
    print("  Posterior policy:")
    for a, prob in enumerate(new_policies[0]):
        action_name = "left" if a == 0 else "right"
        print(f"    {action_name}: {prob:.6f}")

    # 注意：由于我们的补丁，子节点 't' 的策略也会被推理，但并未直接输出。
    # 可以额外计算 t 的值（通过再次调用 search 或从内部数据获取）
    # 这里我们为了验证，额外搜索 't' 状态
    print("\nState 't' (单独验证):")
    root_states_t = ['t']
    new_policies_t, new_values_t = rmcts.search(root_states_t)
    print(f"  Estimated value: {new_values_t[0]:.6f}")
    print("  Posterior policy:")
    for a, prob in enumerate(new_policies_t[0]):
        action_name = "left" if a == 0 else "right"
        print(f"    {action_name}: {prob:.6f}")


if __name__ == "__main__":
    main()