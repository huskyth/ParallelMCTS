import os
import sys
os.chdir('./build')
sys.path.insert(0, '.')   # 将当前目录加入路径
import wmchess.game as game     # 通过包名导入


print(game.numActions())