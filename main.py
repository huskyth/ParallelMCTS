from trainer.train import Trainer
from utils.logger import Logger
from datetime import datetime
from trainer.train_config import TrainConfig
import sys
import traceback

sys.stdout = Logger()

if __name__ == '__main__':
    print(f"🍬 Start logging {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    tn_cfg = TrainConfig()
    t = Trainer(train_config=tn_cfg)
    try:
        t.learn()
    except Exception as err:
        print(err)
        print(traceback.format_exc())
        raise err
