from trainer.train import Trainer
from utils.logger import Logger
from datetime import datetime
from trainer.train_config import TrainConfig
import sys
import traceback
import argparse
import torch
torch.multiprocessing.set_sharing_strategy('file_system')

sys.stdout = Logger()

if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    parser = argparse.ArgumentParser()
    parser.add_argument('--number_of_self_play', type=int, default=5)
    parser.add_argument('--number_of_contest', type=int, default=5)
    print(f"🍬 Start logging {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    args = parser.parse_args()
    tn_cfg = TrainConfig()
    print(f"🍹 执行{args.number_of_self_play}次自我对弈，{args.number_of_contest}次比赛")

    t = Trainer(train_config=tn_cfg, number_of_contest=args.number_of_contest,
                number_of_self_play=args.number_of_self_play)
    try:
        t.learn()
    except Exception as err:
        print(err)
        print(traceback.format_exc())
        raise err
