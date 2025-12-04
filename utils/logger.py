import sys
import time
from constants import ROOT_PATH

log_path = ROOT_PATH / "logs"
if not log_path.exists():
    log_path.mkdir()


class Logger:
    def __init__(self):
        self.terminal = sys.stdout
        self.file = open(log_path / f"app.log", "w", encoding="utf-8")

    def write(self, text):
        self.terminal.write(text)
        self.file.write(text)
        self.flush()

    def flush(self):
        self.terminal.flush()
        self.file.flush()
