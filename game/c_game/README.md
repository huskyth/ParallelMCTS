### 编译
gcc -shared -fPIC -O2 -o librmcts.so rmcts.c game.c random.c -lm -lpthread