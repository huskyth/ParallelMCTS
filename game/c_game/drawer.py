import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

POINTPOS = [
    [0.4017241379310345, 0.06206896551724138],
    [0.3017241379310345, 0.07586206896551724],
    [0.40344827586206894, 0.19482758620689655],
    [0.5, 0.07586206896551724],
    [0.06379310344827586, 0.3137931034482759],
    [0.16896551724137931, 0.4103448275862069],
    [0.04827586206896552, 0.41206896551724137],
    [0.06379310344827586, 0.5172413793103449],
    [0.7396551724137931, 0.31551724137931036],
    [0.6362068965517241, 0.4086206896551724],
    [0.7517241379310344, 0.4068965517241379],
    [0.7379310344827587, 0.5155172413793103],
    [0.40344827586206894, 0.6448275862068965],
    [0.2913793103448276, 0.746551724137931],
    [0.496551724137931, 0.75],
    [0.4017241379310345, 0.7620689655172413],
    [0.4, 0.3137931034482759],
    [0.3, 0.4103448275862069],
    [0.5051724137931034, 0.41206896551724137],
    [0.4051724137931034, 0.5137931034482759],
    [0.4017241379310345, 0.4103448275862069]
]

def render_watermelon_board(state, title="Watermelon Chess", save_path=None, show=True):
    """
    state: numpy array (game_len,)
    title: 标题
    save_path: 如果提供，保存图片到该路径
    show: 是否显示（若无GUI则自动禁用）
    """
    BLACK = 1
    WHITE = -1

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.axis('off')

    radius = 0.03  # 归一化半径

    # 绘制每个点
    for i, (x, y) in enumerate(POINTPOS):
        color_val = state[i+1]
        if color_val == BLACK:
            color = 'black'
        elif color_val == WHITE:
            color = 'red'   # 或 'white' 但边框需要
        else:
            color = 'lightgray'
        circle = plt.Circle((x, y), radius, color=color, ec='gray' if color != 'black' else 'white', linewidth=1)
        ax.add_patch(circle)

    # 显示当前玩家
    player = state[0]
    player_text = "Black's turn" if player == 1 else "White's turn"
    ax.text(0.05, 0.95, player_text, transform=ax.transAxes, fontsize=12, verticalalignment='top')

    plt.title(title)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"图像已保存至 {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)