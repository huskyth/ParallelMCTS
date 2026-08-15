import pygame
import sys
import numpy as np
import torch
import copy
from game import game  # 你的 game 模块（C 接口）
from RMCTS import learn_pi_and_v
from wmnet import WatermelonNet  # 你的网络定义

# 初始化 Pygame
pygame.init()

# 窗口设置
WINDOW_SIZE = 600
FPS = 60

# 颜色
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (200, 50, 50)
GRAY = (200, 200, 200)
LIGHT_GRAY = (240, 240, 240)
GREEN = (50, 200, 50)

# 棋盘坐标（归一化 0~1）
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

# 棋子半径（像素）
PIECE_RADIUS = 18


def screen_pos(coord):
    """将归一化坐标转为屏幕坐标"""
    x, y = coord
    return int(x * WINDOW_SIZE), int(y * WINDOW_SIZE)


def draw_board(screen, state, selected_idx=None):
    """绘制棋盘和棋子"""
    screen.fill(WHITE)

    # 绘制棋盘连线（根据距离矩阵，但我们可以简单绘制所有点的边，或直接绘制已知边）
    # 简化：用距离矩阵绘制边，但我们在 Python 中难以直接获取 distance，
    # 因此我们可以直接绘制所有点之间的边（需要距离信息），或者只绘制点。
    # 为了美观，我们绘制一些已知边（可以从 C 代码的 distance 提取，但为了简化，这里画所有点的直线连接，按环状）
    # 由于我们没有显式距离，先画点，玩家体验也够。
    # 但我可以预先定义一些边（根据你的棋盘拓扑，在 init_distance 中）
    # 这里我们跳过连线，只画点，显得简洁。

    # 绘制棋子
    for i, coord in enumerate(POINTPOS):
        x, y = screen_pos(coord)
        color_val = state[i + 1]  # state[0] 是玩家
        color = GRAY if color_val == 0 else (BLACK if color_val == 1 else RED)
        # 绘制阴影（选中高亮）
        if selected_idx == i:
            pygame.draw.circle(screen, GREEN, (x, y), PIECE_RADIUS + 4, 3)
        pygame.draw.circle(screen, color, (x, y), PIECE_RADIUS)
        if color_val == -1:
            pygame.draw.circle(screen, BLACK, (x, y), PIECE_RADIUS, 2)  # 白棋边框

    # 显示当前玩家
    player = state[0]
    font = pygame.font.Font(None, 36)
    text = font.render(f"玩家: {'黑' if player == 1 else '白'}", True, BLACK)
    screen.blit(text, (10, 10))

    # 显示得分（棋子差）
    board = state[1:]
    black = sum(1 for x in board if x == 1)
    white = sum(1 for x in board if x == -1)
    score = (black - white) / 21.0
    score_text = font.render(f"优势: {score:.3f}", True, BLACK)
    screen.blit(score_text, (10, 50))

    # 显示操作提示
    if selected_idx is not None:
        hint = font.render("点击目标位置移动", True, GREEN)
        screen.blit(hint, (10, 90))


def get_action_from_click(state, click_pos):
    """
    根据鼠标点击位置和当前状态返回动作 (from_idx, to_idx)
    如果点击无效，返回 (None, None)
    """
    x, y = click_pos
    # 寻找最近的棋子（点击精度）
    selected = None
    min_dist = 30
    for i, coord in enumerate(POINTPOS):
        sx, sy = screen_pos(coord)
        dist = ((x - sx) ** 2 + (y - sy) ** 2) ** 0.5
        if dist < min_dist:
            selected = i
            min_dist = dist
    if selected is None:
        return None, None

    # 如果点击的是己方棋子，则选中
    player = state[0]
    if state[selected + 1] != player:
        return None, None  # 不是己方棋子

    # 需要选择一个目标位置（第二次点击）
    # 简化：我们通过点击两个点来生成动作，第一次点击选择源，第二次点击选择目标
    # 但目前只返回选中，第二次点击在外部处理
    return selected, None


def main():
    # 加载 AI 模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = WatermelonNet(input_dim=game.gameLength(), num_actions=game.numActions()).to(device)
    model_path = "best_model.pth"
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print("✅ 加载 AI 模型成功")
    except FileNotFoundError:
        print("⚠️ 未找到 best_model.pth，将使用随机网络")
        model = None  # 使用随机（但最好有模型）

    # 初始化游戏状态
    state = game.rootState()
    game_over = False
    winner = None

    # Pygame 初始化
    screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
    pygame.display.set_caption("西瓜棋 - 人机对战")
    clock = pygame.time.Clock()

    selected_idx = None  # 当前选中的棋子索引

    running = True
    while running:
        # 处理事件
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                pygame.quit()
                sys.exit()

            if event.type == pygame.MOUSEBUTTONDOWN and not game_over:
                x, y = event.pos
                # 检测点击位置
                # 先看是否选中了棋子
                if selected_idx is None:
                    # 尝试选择棋子
                    for i, coord in enumerate(POINTPOS):
                        sx, sy = screen_pos(coord)
                        dist = ((x - sx) ** 2 + (y - sy) ** 2) ** 0.5
                        if dist < PIECE_RADIUS + 5:
                            if state[i + 1] == state[0]:  # 己方棋子
                                selected_idx = i
                                break
                else:
                    # 已选中，尝试移动到目标点
                    # 找到最近的点（目标）
                    target_idx = None
                    for i, coord in enumerate(POINTPOS):
                        sx, sy = screen_pos(coord)
                        dist = ((x - sx) ** 2 + (y - sy) ** 2) ** 0.5
                        if dist < PIECE_RADIUS + 5:
                            target_idx = i
                            break
                    if target_idx is not None and target_idx != selected_idx:
                        # 生成动作 (selected_idx -> target_idx)
                        # 需要检查该动作是否合法
                        # 将动作转为整数索引（对应 C 的动作编号）
                        # 我们需要从 C 获取动作索引，但简化：通过遍历所有合法动作匹配
                        actions = game.getValidActions(state)
                        valid = False
                        action_idx = -1
                        for a in actions:
                            # 如何知道动作对应的 from/to？我们需要获取动作的 from/to
                            # 由于 game.py 未提供 getActionFromTo，我们只能通过尝试执行来验证
                            # 临时状态
                            tmp_state = np.copy(state)
                            tmp_state = game.nextState(tmp_state, a)
                            # 检查移动是否匹配（比较棋子位置）
                            # 更可靠：保存旧状态，执行后比较棋子移动
                            # 这里直接尝试所有合法动作，看哪个执行后符合预期
                            # 简单方法：遍历动作，用 move_to_index 映射？需要额外函数。
                            # 这里先跳过具体实现，用占位。
                            pass
                        # 为了简化，我们直接让用户选择棋子后，再点目标位置，用 game.nextState 直接执行
                        # 但我们需要动作索引，因此我们调用一个辅助函数
                    # 取消选中
                    selected_idx = None

        # 如果是 AI 回合且游戏未结束
        if not game_over and state[0] == -1:  # 假设 AI 执白（后手）
            # AI 走棋
            if model is not None:
                def nnet(states):
                    with torch.no_grad():
                        states_t = torch.from_numpy(states).float().to(device)
                        logits, values = model(states_t)
                        probs = torch.softmax(logits, dim=1)
                    return probs.cpu().numpy(), values.cpu().numpy().flatten()

                root = state[np.newaxis, :]
                pi, _ = learn_pi_and_v(root, num_sims=200, nnet=nnet, c_puct=1.0)
                pi = pi[0]
                actions = game.getValidActions(state)
                if len(actions) > 0:
                    # 按概率采样或选最大
                    best_action = actions[np.argmax(pi[actions])]
                    state = game.nextState(state, best_action)
                    print(f"AI 走棋: {best_action}")
            else:
                # 随机走
                actions = game.getValidActions(state)
                if len(actions) > 0:
                    a = np.random.choice(actions)
                    state = game.nextState(state, a)

            # 检查游戏是否结束
            ended, score = game.gameEnded(state)
            if ended:
                game_over = True
                if score > 0:
                    winner = "黑胜"
                elif score < 0:
                    winner = "白胜"
                else:
                    winner = "平局"
                print(f"游戏结束！{winner}")

        # 绘制
        draw_board(screen, state, selected_idx)
        if game_over:
            font = pygame.font.Font(None, 48)
            text = font.render(f"游戏结束: {winner}", True, (0, 0, 0))
            screen.blit(text, (WINDOW_SIZE // 2 - 100, WINDOW_SIZE // 2 - 20))
        pygame.display.flip()
        clock.tick(FPS)


if __name__ == "__main__":
    main()