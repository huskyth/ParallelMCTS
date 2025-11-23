from pathlib import Path

ROOT_PATH = Path(__file__).parent

cpt = ROOT_PATH / "checkpoints"
if not cpt.exists():
    cpt.mkdir()
