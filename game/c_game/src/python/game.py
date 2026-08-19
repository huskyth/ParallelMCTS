import numpy as np
import random
import numpy.ctypeslib as npct
import ctypes
from pathlib import Path

from . import metaparm

c_int = ctypes.c_int
c_float = ctypes.c_float
array_int32 = npct.ndpointer(dtype=np.int32,ndim=1,flags='CONTIGUOUS')
array_float = npct.ndpointer(dtype=np.float32,ndim=1,flags='CONTIGUOUS')
ptr_int = ctypes.POINTER(c_int)
ptr_float = ctypes.POINTER(c_float)

libgame_path = Path(__file__).parent / 'libgame.so'
libgame = npct.load_library(libgame_path,".")

libnetwork_path = Path(__file__).parent / 'libnetwork.so'
libnetwork = npct.load_library(libnetwork_path,".")

libgame.numActions.restype = c_int
libgame.numActions.argtypes = []

libgame.gameLength.restype = c_int
libgame.gameLength.argtypes = []

libgame.inputLength.restype = c_int
libgame.inputLength.argtypes = []

libgame.rootState.restype = None
libgame.rootState.argtypes = [array_float]

libgame.inputNetwork.restype = None
libgame.inputNetwork.argtypes = [array_float, array_float]

libgame.playerId.restype = c_float
libgame.playerId.argtypes = [array_float]

libgame.gameEnded.restype = c_int
libgame.gameEnded.argtypes = [ptr_float,array_float]

libgame.isValidAction.restype = int
libgame.isValidAction.argtypes = [array_float, c_int]

libgame.getValidActions.restype = c_int
libgame.getValidActions.argtypes = [array_int32,array_float]

libgame.nextState.restype = c_int
libgame.nextState.argtypes = [array_float,array_float,c_int]

libgame.printGame.restype = None
libgame.printGame.argtypes = [array_float]

libnetwork.games_to_nnet_inputs.restype = None
libnetwork.games_to_nnet_inputs.argtypes = [array_float,array_float,c_int]

# 更新 argtypes（在文件开头设置函数原型的部分）
libgame.nextState.restype = c_int
libgame.nextState.argtypes = [array_float, array_float, c_int, ctypes.POINTER(ctypes.c_int)]  # 新增 captures 指针


def numActions():
    return libgame.numActions()

def gameLength():
    return libgame.gameLength()

def inputLength():
    return libgame.inputLength()

def rootState():
    g = np.zeros(gameLength(),dtype=np.float32)
    libgame.rootState(g)
    return g

def inputNetwork(g):
    x = np.zeros(np.prod(metaparm.input_shape), dtype=np.float32)
    libgame.inputNetwork(x,g)
    return x.reshape(metaparm.input_shape)

def inputNetworkMany(G):
    G = np.array(G, dtype=np.float32)
    m = G.shape[0]
    input_len = np.prod(metaparm.input_shape)
    assert input_len == libgame.inputLength()
    X = np.zeros(m*input_len, dtype=np.float32)
    libnetwork.games_to_nnet_inputs(X,G.ravel(),m)
    return X.reshape((m,) + metaparm.input_shape)

def playerId(g):
    if not isinstance(g, np.ndarray):
        g = np.array(g, dtype=np.float32)
    return libgame.playerId(g)

def gameEnded(g):
    if not isinstance(g, np.ndarray):
        g = np.array(g, dtype=np.float32)
    terminal_score = c_float(0.0)
    ended = libgame.gameEnded(ctypes.byref(terminal_score),g)
    return ended, terminal_score.value

def isValidAction(g,a):
    return libgame.isValidAction(g,a)

def getValidActions(g):
    if not isinstance(g, np.ndarray):
        g = np.array(g, dtype=np.float32)
    actions = np.zeros(numActions(),dtype=np.int32)
    num_actions = libgame.getValidActions(actions,g)
    return actions[:num_actions]


def nextState(g, a, captures=None):
    """
    执行动作，返回 (新状态, 吃子数)
    captures: 可选，如果传入 ctypes.c_int 对象，则填充吃子数
    """
    if not isinstance(g, np.ndarray):
        g = np.array(g, dtype=np.float32)
    c_a = c_int(a)
    ga = np.zeros(gameLength(), dtype=np.float32)

    if captures is None:
        # 若未传入 captures，内部创建临时变量并丢弃
        cap = ctypes.c_int()
        res = libgame.nextState(ga, g, c_a, ctypes.byref(cap))
        return ga, cap.value
    else:
        # 若传入，则填充
        res = libgame.nextState(ga, g, c_a, ctypes.byref(captures))
        return ga

def printGame(g):
    libgame.printGame(g)

def randomRollout(g=None, verbose=False):
    game_states = []
    if g is None:
        g = rootState()
    if verbose:
        printGame(g)
        print()
    game_states.append(g)
    ended,terminal_score = gameEnded(g)
    while not ended:
        actions = getValidActions(g)
        a = random.choice(actions)
        g = nextState(g,a)
        if verbose:
            print(f"action {a}:")
            printGame(g)
            print()
        ended,terminal_score = gameEnded(g)
        game_states.append(g)
    return game_states
