import torch
import numpy as np
from .wmnet_gcn import WatermelonGCN
from . import game

def infer(model_path, state=None, top_k=5):
    """
    加载模型，对给定状态进行推理，返回策略概率和价值。
    state: 长度为 22 的 numpy 数组（或 None，默认使用初始局面）
    top_k: 打印概率最高的前 k 个动作
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载网络
    net = WatermelonGCN().to(device)
    net.load_state_dict(torch.load(model_path, map_location=device), strict=False)
    net.eval()

    # 如果没有提供状态，使用默认开局
    if state is None:
        state = game.rootState()
    else:
        state = np.array(state, dtype=np.float32)

    state = state
    # 转换为 Tensor 并推理
    state_t = torch.from_numpy(state).float().unsqueeze(0).to(device)
    with torch.no_grad():
        logits, value = net(state_t)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        value = value.cpu().item()

    # 打印结果
    print(f"📊 当前状态 state = {state}")
    print(f"📊 局面估值 (当前玩家视角): {value:.4f}")
    print(f"📈 策略概率分布 (Top {top_k}):")
    top_indices = np.argsort(probs)[-top_k:][::-1]
    for idx, a in enumerate(top_indices):
        print(f"  {idx+1}. 动作 {a}: {probs[a]:.4f}")

    return probs, value


if __name__ == "__main__":
    # 示例：使用默认开局进行推理
    probs, value = infer("best_model.pth")

    # 你也可以手动构造一个状态（例如从某个棋盘快照）
    # 示例：假设有一个长度为22的状态数组，复制下面的格式
    # custom_state = np.array([1.0, 1,0, -1, ...], dtype=np.float32)
    # infer("best_model.pth", custom_state)