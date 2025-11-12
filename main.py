from train import Trainer
from utils.logger import Logger
from datetime import datetime
from train_config import TrainConfig
import sys

sys.stdout = Logger()

if __name__ == '__main__':
    print(f"🍬 Start logging {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    tn_cfg = TrainConfig()
    t = Trainer(train_config=tn_cfg)
    try:
        t.learn()
    except Exception as err:
        print(err)
