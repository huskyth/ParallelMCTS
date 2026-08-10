import matplotlib.pyplot as plt
import math

def compare_distributions(dist1, dist2,
                          labels=None,
                          title1='分布1',
                          title2='分布2',
                          figsize=(12, 5),
                          color1='#4A90E2',
                          color2='#E27C4A',
                          show_values=True,
                          prob_round=2,
                          show_entropy=True,
                          entropy_base=2):
    """
    将两个概率分布（数组）绘制成一行两列的柱状图进行对比。
    自动归一化（如果数组之和不等于1），并计算各自熵。

    参数:
        dist1, dist2 (list): 两个概率分布数组（长度可以不同，但若需要对比建议长度相同）。
        labels (list): 类别标签，长度应与较长的数组一致，或自动生成索引。
        title1, title2 (str): 左右子图标题。
        figsize: 画布大小。
        color1, color2: 柱状图颜色。
        show_values: 是否在柱顶显示数值。
        prob_round: 概率保留小数位数。
        show_entropy: 是否显示熵。
        entropy_base: 熵的底数，默认2（比特）。
    """
    # ---------- 数据校验与归一化 ----------
    def normalize(arr):
        total = sum(arr)
        if total == 0:
            raise ValueError("数组和为0，无法归一化")
        return [x / total for x in arr]

    p1 = normalize(dist1)
    p2 = normalize(dist2)

    # 处理标签：取较长的长度，如果未提供标签则自动生成
    max_len = max(len(p1), len(p2))
    if labels is None:
        labels = list(range(1, max_len + 1))
    else:
        if len(labels) < max_len:
            # 补齐
            labels = list(labels) + list(range(len(labels)+1, max_len+1))
        else:
            labels = labels[:max_len]  # 截断

    # 确保两个分布长度一致（补齐0）
    if len(p1) < max_len:
        p1 = p1 + [0] * (max_len - len(p1))
    if len(p2) < max_len:
        p2 = p2 + [0] * (max_len - len(p2))

    # ---------- 计算熵 ----------
    def entropy(prob_list, base):
        return -sum(p * math.log(p, base) for p in prob_list if p > 0)

    if show_entropy:
        h1 = entropy(p1, entropy_base)
        h2 = entropy(p2, entropy_base)
        unit = "bits" if entropy_base == 2 else "nats"
        # 在总标题展示两个熵的对比
        suptitle = f"\n\n熵对比：{title1} = {h1:.4f} {unit}  |  {title2} = {h2:.4f} {unit}"
    else:
        suptitle = ""

    # ---------- 中文配置 ----------
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'PingFang SC', 'WenQuanYi Micro Hei']
    plt.rcParams['axes.unicode_minus'] = False

    # ---------- 绘图 ----------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    if suptitle:
        fig.suptitle(suptitle, fontsize=14, fontweight='bold', y=1.02)

    # 左图：分布1
    bars1 = ax1.bar(labels, p1, color=color1)
    ax1.set_title(title1, fontsize=14)
    ax1.set_xlabel('类别')
    ax1.set_ylabel('概率')
    if show_values:
        for bar in bars1:
            h = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., h + 0.01,
                     f'{h:.{prob_round}f}', ha='center', va='bottom', fontsize=10)

    # 右图：分布2
    bars2 = ax2.bar(labels, p2, color=color2)
    ax2.set_title(title2, fontsize=14)
    ax2.set_xlabel('类别')
    ax2.set_ylabel('概率')
    if show_values:
        for bar in bars2:
            h = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., h + 0.01,
                     f'{h:.{prob_round}f}', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.show()
    return fig, (ax1, ax2)