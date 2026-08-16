import json
import os

import sys

import numpy as np

from game.chess.chess import Chess

rp = r"C:\Users\qq162\Desktop\ParallelMCTS\game\c_game\trajectories"
c = Chess()

for fname in os.listdir(rp):
    if fname.endswith(".json"):
        with open(f"{rp}/{fname}") as f:
            data = json.load(f)
        print(f"轨迹: {fname}, 结束原因: {data['reason']}, {len(data['history'])} 步")
        print("前3步的信息:")
        for i, h in enumerate(data['history']):
            c.pointStatus = h['state'][1:]
            assert len(h['state'][1:]) == 21
            pi = h['policy']

            c.image_show("a", True, 0, pi=pi, player=h['state'][0], step_reward=h['step_reward'],
                         return_val=h['return'])
            print(
                f"  步 {i}: state前5个值={h['state']}, step_reward={h['step_reward']}, return={h['return']}, {h['policy']}, {np.argmax(np.array(h['policy']))}")
        break  # 只打印第一个文件
