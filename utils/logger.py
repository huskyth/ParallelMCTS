import sys
import time
from pathlib import Path

ROOT_PATH = Path(__file__).parent.parent
log_path = Path(ROOT_PATH / "logs")
if not log_path.exists():
    log_path.mkdir()


class Logger:
    def __init__(self):
        self.terminal = sys.stdout
        time_ms = time.time()
        self.file = open(log_path / f"log_{time_ms}.txt", "w", encoding="utf-8")

    def write(self, text):
        self.terminal.write(text)
        self.file.write(text)

    def flush(self):
        self.terminal.flush()
        self.file.flush()
