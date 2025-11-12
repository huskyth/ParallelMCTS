from train import Trainer
from utils.logger import Logger
from datetime import datetime
import sys

sys.stdout = Logger()

if __name__ == '__main__':
    print(f"🍬 Start logging {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    t = Trainer()
    t.learn()
