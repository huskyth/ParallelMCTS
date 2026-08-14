import numpy as np
from . import game
from .RMCTS import learn_pi_and_v

def build_eat_state():
    return np.array([-1, 0, 0, 0, -1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, -1, 0, 0, 0, -1, 0, 0], dtype=np.float32) * -1


def rmcts_eat():
    state = build_eat_state()
    print("初始棋盘（棋子状态，索引1~21）:")
    print(state[1:])

    # 获取所有合法动作
    actions = game.getValidActions(state)
    print(f"合法动作数: {len(actions)}")
    if len(actions) == 0:
        print("⚠️ 无合法动作，请检查构造的局面。")
        return

    # 指定吃子动作（你需要根据规则找出哪个动作是吃子）
    # 方法1：如果你知道动作索引，直接赋值
    # eat_action = 42  # 请替换为实际索引

    # 方法2：自动检测吃子（如果吃子会导致棋子数减少）
    # 我们遍历所有合法动作，执行后看棋子数是否减少。
    initial_count = np.count_nonzero(state[1:] != 0)
    eat_action = None
    for a in actions:
        tmp_state = np.copy(state)
        tmp_state = game.nextState(tmp_state, a)  # 注意：nextState 返回新状态
        new_count = np.count_nonzero(tmp_state[1:] != 0)
        if new_count < initial_count:
            eat_action = a
            break

    if eat_action is None:
        print("⚠️ 未检测到任何吃子动作，请检查构造的局面是否符合吃子规则。")
        return
    else:
        print(f"检测到吃子动作索引: {eat_action}")

    # 使用均匀先验（无任何先验知识）
    def uniform_nnet(states):
        n = game.numActions()
        batch = states.shape[0]
        pi = np.ones((batch, n), dtype=np.float32) / n
        v = np.zeros(batch, dtype=np.float32)
        return pi, v

    # 调用 RMCTS
    root = state[np.newaxis, :]
    pi, _ = learn_pi_and_v(root, numSims=400, nnet=uniform_nnet, c_puct=2.0)
    pi = pi[0]

    # 找出 RMCTS 推荐的最优动作
    best_action = np.argmax(pi)
    # 查看吃子动作的概率
    eat_prob = pi[eat_action]
    best_prob = pi[best_action]

    print("\n=== RMCTS 搜索结果 ===")
    print(f"吃子动作索引: {eat_action}, 概率: {eat_prob:.4f}")
    print(f"最优动作索引: {best_action}, 概率: {best_prob:.4f}")

    print(pi)

    if best_action == eat_action:
        print("✅ RMCTS 成功选择吃子动作！")
    else:
        print("❌ RMCTS 未选择吃子动作。尝试增加 num_sims 或调整 c_puct。")
        # 可选：打印 top 5 动作概率
        top5 = np.argsort(pi)[-5:][::-1]
        print("Top 5 动作及其概率:")
        for i, idx in enumerate(top5):
            print(f"  {i+1}: 动作 {idx}, 概率 {pi[idx]:.4f}")

def main():
    rmcts_eat()