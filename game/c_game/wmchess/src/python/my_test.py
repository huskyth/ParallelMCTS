import torch
import numpy as np
from .RMCTS import learn_pi_and_v
from .wmnet import WatermelonNet  # 你的网络类
import torch.optim as optim
from . import game
def build_eat_state():
    return np.array([-1, 0, 0, 0, -1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, -1, 0, 0, 0, -1, 0, 0], dtype=np.float32) * -1

def test_single_step_learning():
    # 1,构造那个“一步吃子”的状态（复用你之前的 build_eat_state）
    state = build_eat_state()
    root = state[np.newaxis, :]

    # 2,用 RMCTS 生成目标（这是一个“好标签”）
    def mock_or_real_nnet(states):
        # 注意：这里为了纯粹测试网络拟合能力，我们不用真实网络生成先验，用均匀先验
        # 但如果你当前有网络，也可以用它，这里为了排除干扰，直接用均匀先验让 RMCTS 找出吃子
        n = game.numActions()
        pi = np.ones((states.shape[0], n)) / n
        v = np.zeros(states.shape[0])
        return pi.astype(np.float32), v.astype(np.float32)

    target_pi, _ = learn_pi_and_v(root, numSims=400, nnet=mock_or_real_nnet, c_puct=2.0)
    target_pi = target_pi[0]  # 这是一个“告诉网络要往吃子方向走”的分布
    eat_action = np.argmax(target_pi)  # 理论上应该是吃子动作

    # 3,初始化一个随机网络，并进行一次梯度下降
    net = WatermelonNet(input_dim=game.gameLength(), num_actions=game.numActions())
    optimizer = optim.Adam(net.parameters(), lr=0.001)

    # 记录更新前的概率
    with torch.no_grad():
        logits_before = net(torch.FloatTensor(state).unsqueeze(0))
        prob_before = torch.softmax(logits_before[0], dim=1)[0, eat_action].item()

    # 执行一次前向 + 反向
    for _ in range(57):
        optimizer.zero_grad()
        logits = net(torch.FloatTensor(state).unsqueeze(0))
        loss = -torch.mean(torch.sum(torch.tensor(target_pi).unsqueeze(0) * torch.log_softmax(logits[0], dim=1), dim=1))
        loss.backward()
        optimizer.step()

    # 记录更新后的概率
    with torch.no_grad():
        logits_after = net(torch.FloatTensor(state).unsqueeze(0))
        prob_after = torch.softmax(logits_after[0], dim=1)[0, eat_action].item()

    print(f"更新前吃子动作概率: {prob_before:.4f}")
    print(f"更新后吃子动作概率: {prob_after:.4f}")
    print(f"概率提升: {prob_after - prob_before:.4f}")

    if prob_after - prob_before > 0.01:
        print("✅ 网络学习正常，梯度有效传递。")
    else:
        print("🚨 网络几乎没动！请检查学习率是否太小（试试0.005）或梯度是否被裁剪了。")