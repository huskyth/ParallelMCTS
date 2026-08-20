import pygame
import sys
import numpy as np
import torch
from . import game
from .RMCTS import learn_pi_and_v
from .wmnet_gcn import WatermelonGCN
from .wmnet_not_use import WatermelonNet

pygame.init()

WINDOW_SIZE = 700
FPS = 60

# 颜色
COLOR_BG = (18, 25, 45)
COLOR_BOARD_LINE = (100, 130, 180)
COLOR_TEXT = (220, 230, 255)
COLOR_TEXT_HIGHLIGHT = (100, 220, 255)
COLOR_BLACK_PIECE = (15, 15, 25)
COLOR_WHITE_PIECE = (220, 220, 230)
COLOR_EMPTY = (60, 70, 95)
COLOR_SELECTED = (100, 220, 255)

# 角色定义
HUMAN_COLOR = -1   # 白棋
AI_COLOR = 1       # 黑棋

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

PIECE_RADIUS = 22
DOT_RADIUS = 4
CLICK_RADIUS = PIECE_RADIUS + 25

def screen_pos(coord, offset=0):
    margin = 80
    size = WINDOW_SIZE - 2 * margin
    x = margin + coord[0] * size
    y = margin + coord[1] * size
    return int(x + offset), int(y + offset)

def draw_board(screen, state, selected_idx=None, thinking=False, game_over=False, winner=None):
    screen.fill(COLOR_BG)

    # 装饰网格
    for i, p1 in enumerate(POINTPOS):
        for j, p2 in enumerate(POINTPOS):
            if i < j:
                dx = p1[0] - p2[0]
                dy = p1[1] - p2[1]
                if dx*dx + dy*dy < 0.04:
                    x1, y1 = screen_pos(p1)
                    x2, y2 = screen_pos(p2)
                    pygame.draw.line(screen, COLOR_BOARD_LINE, (x1, y1), (x2, y2), 1)

    for i, coord in enumerate(POINTPOS):
        x, y = screen_pos(coord)
        color_val = state[i + 1]
        if selected_idx == i:
            pygame.draw.circle(screen, COLOR_SELECTED, (x, y), PIECE_RADIUS + 6, 3)
        if color_val == 0:
            pygame.draw.circle(screen, COLOR_EMPTY, (x, y), DOT_RADIUS)
        else:
            shadow_offset = 3
            pygame.draw.circle(screen, (0,0,0), (x+shadow_offset, y+shadow_offset), PIECE_RADIUS)
            color = COLOR_BLACK_PIECE if color_val == 1 else COLOR_WHITE_PIECE
            pygame.draw.circle(screen, color, (x, y), PIECE_RADIUS)
            if color_val == 1:
                highlight_color = (60, 60, 80)
            else:
                highlight_color = (255, 255, 255)
            pygame.draw.circle(screen, highlight_color, (x-4, y-4), PIECE_RADIUS//3)

    font = pygame.font.Font(None, 30)
    player = state[0]
    if thinking:
        status = "AI is thinking..."
    else:
        # 现在人类是白棋（-1），AI是黑棋（1）
        if player == HUMAN_COLOR:
            status = "Your turn (White)"
        elif player == AI_COLOR:
            status = "AI's turn (Black)"
        else:
            status = ""
    status_color = COLOR_TEXT_HIGHLIGHT if player == HUMAN_COLOR else COLOR_TEXT
    text = font.render(status, True, status_color)
    screen.blit(text, (20, WINDOW_SIZE - 50))

    board = state[1:]
    black = sum(1 for x in board if x == 1)
    white = sum(1 for x in board if x == -1)
    score = (black - white) / 21.0
    score_text = font.render(f"Black: {black}  White: {white}  Advantage: {score:.3f}", True, COLOR_TEXT)
    screen.blit(score_text, (20, WINDOW_SIZE - 80))

    if game_over and winner is not None:
        font_big = pygame.font.Font(None, 60)
        text = font_big.render(f"Game Over: {winner}", True, (255, 215, 0))
        screen.blit(text, (WINDOW_SIZE//2 - text.get_width()//2, WINDOW_SIZE//2 - 30))

def get_action_from_to(state, from_idx, to_idx):
    actions = game.getValidActions(state)
    player = state[0]
    for a in actions:
        test_state = np.copy(state)
        test_state, _ = game.nextState(test_state, a)
        if test_state[from_idx + 1] == 0 and test_state[to_idx + 1] == player:
            return a
    return -1

def start():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = WatermelonNet().to(device)
    model_path = "best_model.pth"
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print("AI model loaded")
    except FileNotFoundError:
        print("best_model.pth not found, using random")
        model = None

    state = game.rootState()  # 初始 player=1 (黑先)
    game_over = False
    winner = None
    thinking = False
    selected_idx = None

    screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
    pygame.display.set_caption("Watermelon Chess - Human (White) vs AI (Black)")
    clock = pygame.time.Clock()

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r and game_over:
                    state = game.rootState()
                    game_over = False
                    winner = None
                    selected_idx = None
                    thinking = False
                if event.key == pygame.K_ESCAPE:
                    running = False
                    pygame.quit()
                    sys.exit()

            # 人类回合：当前玩家是白棋（-1）
            if event.type == pygame.MOUSEBUTTONDOWN and not game_over and not thinking and state[0] == HUMAN_COLOR:
                x, y = event.pos
                clicked_idx = None
                for i, coord in enumerate(POINTPOS):
                    sx, sy = screen_pos(coord)
                    if ((x - sx)**2 + (y - sy)**2)**0.5 < CLICK_RADIUS:
                        clicked_idx = i
                        break

                if selected_idx is None:
                    # 选择己方白棋
                    if clicked_idx is not None and state[clicked_idx + 1] == HUMAN_COLOR:
                        selected_idx = clicked_idx
                else:
                    if clicked_idx is None:
                        selected_idx = None
                    else:
                        if state[clicked_idx + 1] == HUMAN_COLOR:
                            if clicked_idx != selected_idx:
                                selected_idx = clicked_idx
                            else:
                                selected_idx = None
                        elif state[clicked_idx + 1] == 0:
                            if clicked_idx != selected_idx:
                                action = get_action_from_to(state, selected_idx, clicked_idx)
                                if action != -1:
                                    state, _ = game.nextState(state, action)
                                    selected_idx = None
                                    ended, score = game.gameEnded(state)
                                    if ended:
                                        game_over = True
                                        winner = "Black" if score > 0 else ("White" if score < 0 else "Draw")
                                else:
                                    selected_idx = None
                            else:
                                selected_idx = None
                        else:
                            selected_idx = None

        # AI 回合：当前玩家是黑棋（1）
        if not game_over and state[0] == AI_COLOR and not thinking:
            thinking = True
            pygame.time.wait(300)
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
            thinking = False

        draw_board(screen, state, selected_idx, thinking, game_over, winner)
        pygame.display.flip()
        clock.tick(FPS)

if __name__ == "__main__":
    start()