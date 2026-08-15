### 编译
gcc -shared -fPIC -O2 -o librmcts.so rmcts.c game.c random.c -lm -lpthread

### 操作步骤
1. sh build_all_games.sh
2. python run.py

### 添加吃子奖励