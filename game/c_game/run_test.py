# run.py (放在 rmcts/ 根目录)
import sys
sys.path.insert(0, './build')   # 将 build 目录加入 Python 路径

# 从 wmchess 包导入你的训练模块
from wmchess.eva_test import main

if __name__ == "__main__":
    main()