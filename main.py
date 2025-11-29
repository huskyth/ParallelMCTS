from game.abstract_game import AbstractGame
from trainer.train import Trainer
from utils.logger import Logger
from datetime import datetime
from trainer.train_config import TrainConfig
import sys
import traceback
import argparse
import torch

sys.stdout = Logger()

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--number_of_self_play', type=int, default=1)
    parser.add_argument('--number_of_contest', type=int, default=10)
    parser.add_argument('--use_concurrent', type=bool, default=False)
    parser.add_argument('--is_render', type=bool, default=False)
    parser.add_argument('--mode', type=str, default="train")
    parser.add_argument('--game', type=str, default="tictactoe", choices=['WMChess', 'tictactoe'])
    print(f"🍬 Start logging {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    args = parser.parse_args()
    if args.use_concurrent:
        torch.multiprocessing.set_sharing_strategy('file_system')
        torch.multiprocessing.set_start_method('spawn')
    tn_cfg = TrainConfig()
    abs_game = AbstractGame(args.game, is_render=args.is_render)
    print(f"🍹 执行{args.number_of_self_play}次自我对弈，{args.number_of_contest}次比赛")

    t = Trainer(train_config=tn_cfg, number_of_contest=args.number_of_contest,
                number_of_self_play=args.number_of_self_play, abstract_game=abs_game,
                use_pool=args.use_concurrent, is_render=args.is_render)
    try:
        if args.mode == 'train':
            t.learn()
        elif args.mode == "test":
            t.test()
        elif args.mode == "play":
            t.play("AI")
    except Exception as err:
        print(err)
        print(traceback.format_exc())
        raise err
