import pygame
import sys
import numpy as np
import torch
from . import game
from .RMCTS import learn_pi_and_v
from .wmnet_gcn import WatermelonGCN

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

PIECE_RADIUS = 18

def screen_pos(coord):
    """将归一化坐标转为屏幕坐标"""
    x, y = coord
    return int(x * WINDOW_SIZE), int(y * WINDOW_SIZE)

def draw_board(screen, state, selected_idx=None):
    """绘制棋盘和棋子"""
    screen.fill(WHITE)

    # 绘制棋子
    for i, coord in enumerate(POINTPOS):
        x, y = screen_pos(coord)
        color_val = state[i + 1]
        color = GRAY if color_val == 0 else (BLACK if color_val == 1 else RED)
        if selected_idx == i:
            pygame.draw.circle(screen, GREEN, (x, y), PIECE_RADIUS + 4, 3)
        pygame.draw.circle(screen, color, (x, y), PIECE_RADIUS)
        if color_val == -1:
            pygame.draw.circle(screen, BLACK, (x, y), PIECE_RADIUS, 2)

    # 显示玩家（英文避免乱码）
    player = state[0]
    font = pygame.font.Font(None, 36)
    text = font.render(f"Player: {'Black' if player == 1 else 'White'}", True, BLACK)
    screen.blit(text, (10, 10))

    # 显示得分
    board = state[1:]
    black = sum(1 for x in board if x == 1)
    white = sum(1 for x in board if x == -1)
    score = (black - white) / 21.0
    score_text = font.render(f"Score: {score:.3f}", True, BLACK)
    screen.blit(score_text, (10, 50))

    if selected_idx is not None:
        hint = font.render("Click target", True, GREEN)
        screen.blit(hint, (10, 90))

def get_action_from_to(state, from_idx, to_idx):
    """
    根据状态、起点和终点，返回合法的动作索引，如果不存在则返回 -1。
    """
    actions = game.getValidActions(state)
    player = state[0]
    for a in actions:
        test_state = np.copy(state)
        # nextState 返回 (new_state, captures)
        test_state, _ = game.nextState(test_state, a)
        # 检查起点被清空，终点变成玩家棋子（即移动成功）
        if test_state[from_idx + 1] == 0 and test_state[to_idx + 1] == player:
            return a
    return -1

def start():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = WatermelonGCN().to(device)
    model_path = "best_model.pth"
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print("AI model loaded")
    except FileNotFoundError:
        print("best_model.pth not found, using random")
        model = None

    state = game.rootState()
    game_over = False
    winner = None

    screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
    pygame.display.set_caption("Watermelon Chess")
    clock = pygame.time.Clock()

    selected_idx = None

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                pygame.quit()
                sys.exit()

            if event.type == pygame.MOUSEBUTTONDOWN and not game_over:
                x, y = event.pos
                if selected_idx is None:
                    # 选择己方棋子
                    for i, coord in enumerate(POINTPOS):
                        sx, sy = screen_pos(coord)
                        dist = ((x - sx) ** 2 + (y - sy) ** 2) ** 0.5
                        if dist < PIECE_RADIUS + 5:
                            if state[i + 1] == state[0]:
                                selected_idx = i
                                break
                else:
                    # 选择目标位置
                    target_idx = None
                    for i, coord in enumerate(POINTPOS):
                        sx, sy = screen_pos(coord)
                        dist = ((x - sx) ** 2 + (y - sy) ** 2) ** 0.5
                        if dist < PIECE_RADIUS + 5:
                            target_idx = i
                            break
                    if target_idx is not None and target_idx != selected_idx:
                        action = get_action_from_to(state, selected_idx, target_idx)
                        if action != -1:
                            # nextState 返回 (new_state, captures)
                            state, _ = game.nextState(state, action)
                            ended, score = game.gameEnded(state)
                            if ended:
                                game_over = True
                                winner = "Black" if score > 0 else ("White" if score < 0 else "Draw")
                                print(f"Game over! Winner: {winner}")
                            selected_idx = None
                        else:
                            # 非法移动，取消选中
                            selected_idx = None
                    else:
                        # 点击了相同位置或空白，取消选中
                        selected_idx = None

        # AI 回合（白棋）
        if not game_over and state[0] == -1:
            if model is not None:
                def nnet(states):
                    with torch.no_grad():
                        states_t = torch.from_numpy(states).float().to(device)
                        logits, values = model(states_t)
                        probs = torch.softmax(logits, dim=1)
                    return probs.cpu().numpy(), values.cpu().numpy().flatten()

                root = state[np.newaxis, :]
                pi, _ = learn_pi_and_v(root, numSims=200, nnet=nnet, c_puct=1.0)
                pi = pi[0]
                actions = game.getValidActions(state)
                if len(actions) > 0:
                    best_action = actions[np.argmax(pi[actions])]
                    state, _ = game.nextState(state, best_action)
                    print(f"AI move: {best_action}")
            else:
                actions = game.getValidActions(state)
                if len(actions) > 0:
                    a = np.random.choice(actions)
                    state, _ = game.nextState(state, a)

            ended, score = game.gameEnded(state)
            if ended:
                game_over = True
                winner = "Black" if score > 0 else ("White" if score < 0 else "Draw")
                print(f"Game over! Winner: {winner}")

        draw_board(screen, state, selected_idx)
        if game_over:
            font = pygame.font.Font(None, 48)
            text = font.render(f"Game Over: {winner}", True, BLACK)
            screen.blit(text, (WINDOW_SIZE // 2 - 100, WINDOW_SIZE // 2 - 20))
        pygame.display.flip()
        clock.tick(FPS)

if __name__ == "__main__":
    main()