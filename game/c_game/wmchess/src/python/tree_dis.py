from pathlib import Path

import torch

from . import game, RMCTS
from .wmnet_gcn import WatermelonGCN


def ds():
    best_model_path = 'best_model.pth'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = WatermelonGCN().to(device)
    net.load_state_dict(torch.load(best_model_path, map_location=device), strict=False)
    R = RMCTS.RMCTS_Tree(game.rootState(), net)
    T,L = R.explore(1400, temperature=1)
    R.export_tree_json("R_tree_1400.json", include_state=True)

