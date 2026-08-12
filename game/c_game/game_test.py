import ctypes
import numpy as np
import random
from drawer import render_watermelon_board  # 导入可视化函数

# ---------- 加载动态库 ----------
lib = ctypes.CDLL("./librmcts.so")

# 设置函数原型（同上，略，重复之前的设置）
# ---------- 设置所有需要使用的函数原型 ----------
lib.numActions.argtypes = []
lib.numActions.restype = ctypes.c_int

lib.gameLength.argtypes = []
lib.gameLength.restype = ctypes.c_int

lib.inputLength.argtypes = []
lib.inputLength.restype = ctypes.c_int

lib.rootState.argtypes = [ctypes.POINTER(ctypes.c_float)]
lib.rootState.restype = None

lib.playerId.argtypes = [ctypes.POINTER(ctypes.c_float)]
lib.playerId.restype = ctypes.c_float

lib.getValidActions.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_float)]
lib.getValidActions.restype = ctypes.c_int

lib.isValidAction.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_int]
lib.isValidAction.restype = ctypes.c_int

lib.nextState.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.c_int]
lib.nextState.restype = ctypes.c_int

lib.gameEnded.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float)]
lib.gameEnded.restype = ctypes.c_int

lib.printGame.argtypes = [ctypes.POINTER(ctypes.c_float)]
lib.printGame.restype = None

# ---------- 获取游戏参数 ----------
game_len = lib.gameLength()
num_actions = lib.numActions()
print(f"游戏状态长度: {game_len}, 动作总数: {num_actions}")

# ---------- 分配状态并初始化 ----------
state = np.zeros(game_len, dtype=np.float32)
lib.rootState(state.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))

# 显示初始棋盘
print("初始棋盘")
render_watermelon_board(state, title="Step 0")  # 等待按键

# ---------- 模拟对局循环 ----------
move_count = 0
while True:
    player = lib.playerId(state.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))
    print(f"\n第 {move_count+1} 步，当前玩家: {'黑' if player == 1.0 else '白'}")

    # 获取合法动作
    actions = np.zeros(num_actions, dtype=np.int32)
    count = lib.getValidActions(actions.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
                                state.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))
    if count == 0:
        print("无合法动作，游戏结束")
        break

    idx = random.randint(0, count - 1)
    chosen_action = actions[idx]
    print(f"随机选择动作索引: {chosen_action}")

    # 执行动作
    new_state = np.zeros(game_len, dtype=np.float32)
    ret = lib.nextState(new_state.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                        state.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                        chosen_action)
    state = new_state
    move_count += 1

    # 检查是否终局
    score = ctypes.c_float()
    ended = lib.gameEnded(ctypes.byref(score),
                          state.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))

    # 显示当前棋盘
    title = f"Step {move_count}" + (" (终局)" if ended else "")
    render_watermelon_board(state, title=title)  # 每步等待按键

    if ended:
        print("游戏结束!")
        print(f"得分: {score.value}")
        if score.value > 0:
            print("玩家1 (白) 获胜!")
        elif score.value < 0:
            print("玩家2 (黑) 获胜!")
        else:
            print("平局?")
        break