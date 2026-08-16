// test_captures.c
#include <stdio.h>
#include "game.h"

int main() {
    // 初始化棋盘（默认开局）
    float state[22] = {-1, 0, 0, 0, -1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, -1, 0, 0, 0, -1, 0, 0};
    rootState(state);
    printf("初始棋盘:\n");
    printGame(state);

    // 获取合法动作
    int actions[100];
    int num = getValidActions(actions, state);
    printf("合法动作数: %d\n", num);

    // 遍历所有动作，执行并查看吃子数
    for (int i = 0; i < num; i++) {
        int a = actions[i];
        float next[22];
        int captures;
        int ret = nextState(next, state, a, &captures);
        printf("动作 %d: 返回值 %d, 吃子数 %d\n", a, ret, captures);
        // 打印执行后的棋盘（可选）
        // printGame(next);
    }

    return 0;
}