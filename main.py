from trainer.train import Trainer
from utils.logger import Logger
from datetime import datetime
from trainer.train_config import TrainConfig
import sys
import traceback
import argparse

sys.stdout = Logger()

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--number_of_self_play', type=int, default=1)
    parser.add_argument('--number_of_contest', type=int, default=10)
    parser.add_argument('--no_concurrent', action='store_false', default=True)
    parser.add_argument('--is_render', type=bool, default=False)
    parser.add_argument('--no_data_augment', action='store_false', default=True)
    parser.add_argument('--no_swanlab', action='store_false', default=True)
    parser.add_argument('--is_image_show', type=bool, default=False)
    parser.add_argument('--mode', type=str, default="play", choices=['train', 'test', 'play'])
    parser.add_argument('--test_number', type=int, default=10)
    parser.add_argument('--game', type=str, default="WMChess", choices=['WMChess', 'tictactoe'])
    print(f"🍬 Start logging {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    args = parser.parse_args()
    if args.no_concurrent:
        import torch.multiprocessing as mp
        mp.set_start_method('spawn', force=True)

    tn_cfg = TrainConfig()
    print(f"🍹 执行{args.number_of_self_play}次自我对弈，{args.number_of_contest}次比赛")

    t = Trainer(train_config=tn_cfg, number_of_contest=args.number_of_contest,
                number_of_self_play=args.number_of_self_play,
                use_pool=args.no_concurrent, is_render=args.is_render, is_data_augment=args.no_data_augment,
                is_image_show=args.is_image_show, is_continue=False, game=args.game, use_swanlab=args.no_swanlab)
    try:
        if args.mode == 'train':
            t.learn()
        elif args.mode == "test":
            t.test(args.test_number)
        elif args.mode == "play":
            t.play("Human")
    except Exception as err:
        print(err)
        print(traceback.format_exc())
        raise err
