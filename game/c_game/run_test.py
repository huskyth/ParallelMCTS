# run.py (放在 rmcts/ 根目录)
import copy
import sys
sys.path.insert(0, './build')   # 将 build 目录加入 Python 路径
import torch
# 从 wmchess 包导入你的训练模块
# from wmchess.eva_test import main
# from wmchess.vs import evaluate_best_vs_random
# from wmchess.model_vs_pure_random import main

# from wmchess.tree_dis import *
from wmchess.wm_play import start
from wmchess.train import evaluate_vs_previous
#from wmchess.wmnet_gcn import WatermelonGCN
from wmchess.wmnet_not_use import WatermelonNet
# from wmchess.eva_model import main
# from wmchess.infer import infer
# from wmchess.tree_dis import ds
if __name__ == "__main__":
    # s = [1, 1, 0, 0, 0, -1, -1, 0, 1, 0, 0, 1, 0, 0, -1, 0, 0, 0.,
    #      0, 0, 0, 0.]
    # probs, value = infer("best_model.pth",state=s)


    # main()

    start()
    #
    #
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    random_net = WatermelonNet().to(device)
    #
    net = WatermelonNet().to(device)
    #
    #
    best_model_path = 'best_model.pth'
    net.load_state_dict(torch.load(best_model_path, map_location=device))
    win_rate_vs_best = evaluate_vs_previous( net, random_net, 1200, 1, device, num_starts=100)
    print(f"Epoch {0}, Win Rate vs Best: {win_rate_vs_best:.3f}")

