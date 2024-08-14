import os
from torch.utils.tensorboard import SummaryWriter
import wandb

from chess.common import ROOT_PATH, create_directory

WANDB_PATH = ROOT_PATH / "wandb_log"
SUMMARY_PATH = ROOT_PATH / "summary_log"
create_directory(WANDB_PATH)
create_directory(SUMMARY_PATH)

log_dir = SUMMARY_PATH


class MySummary:

    def __init__(self, log_dir_name="default", use_wandb=True):
        log_path = str(log_dir / log_dir_name)
        if not os.path.exists(log_path):
            os.mkdir(log_path)
        self.writer = SummaryWriter(log_dir=log_path)
        self.use_wandb = use_wandb
        if use_wandb:
            wandb.login(key="613f55cae781fb261b18bad5ec25aa65766e6bc8")
            self.wandb_logger = wandb.init(project="Parallel", dir=WANDB_PATH)

    def add_float(self, x, y, title, x_name=None):
        self.writer.add_scalar(title, y, x)
        if self.use_wandb:
            self.wandb_logger.log({title: y})

    def close(self):
        self.writer.close()


my_summary = MySummary(use_wandb=False)

if __name__ == '__main__':
    test = MySummary()
    for i in range(100):
        test.add_float(x=i, y=i ** 2, title="y=f(x)", x_name="x")
