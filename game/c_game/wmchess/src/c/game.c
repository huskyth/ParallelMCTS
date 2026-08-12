#include "game.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

// ---------- 常量 ----------
#define BOARD_SIZE 21
#define BLACK       1
#define WHITE      -1

// ---------- 全局动作列表 ----------
static int NUM_ACTIONS = 0;
static int ACTION_FROM[512];
static int ACTION_TO[512];

// ---------- 距离矩阵 ----------
static int distance[BOARD_SIZE][BOARD_SIZE];

// ---------- 初始化距离矩阵（使用你的 21x21 数据） ----------
static void init_distance() {
    int raw[BOARD_SIZE][BOARD_SIZE] = {
        {0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
        {1,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
        {1,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0},
        {1,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0},
        {0,1,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
        {0,0,0,0,1,0,1,1,0,0,0,0,0,0,0,0,0,1,0,0,0},
        {0,0,0,0,1,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0},
        {0,0,0,0,0,1,1,0,0,0,0,0,0,1,0,0,0,0,0,0,0},
        {0,0,0,1,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0},
        {0,0,0,0,0,0,0,0,1,0,1,1,0,0,0,0,0,0,1,0,0},
        {0,0,0,0,0,0,0,0,1,1,0,1,0,0,0,0,0,0,0,0,0},
        {0,0,0,0,0,0,0,0,0,1,1,0,0,0,1,0,0,0,0,0,0},
        {0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,1,0},
        {0,0,0,0,0,0,0,1,0,0,0,0,1,0,0,1,0,0,0,0,0},
        {0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,1,0,0,0,0,0},
        {0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0},
        {0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,1},
        {0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,1,1},
        {0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,1,1},
        {0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,1,1,0,1},
        {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,0}
    };
    for (int i = 0; i < BOARD_SIZE; i++)
        for (int j = 0; j < BOARD_SIZE; j++)
            distance[i][j] = raw[i][j];
}

// ---------- 初始化动作列表 ----------
static void init_actions() {
    if (NUM_ACTIONS != 0) return;
    init_distance();
    int count = 0;
    for (int from = 0; from < BOARD_SIZE; from++) {
        for (int to = 0; to < BOARD_SIZE; to++) {
            if (distance[from][to] == 1) {
                ACTION_FROM[count] = from;
                ACTION_TO[count] = to;
                count++;
            }
        }
    }
    NUM_ACTIONS = count;
}

// ---------- 辅助函数 ----------
static inline int get_piece(const float* g, int idx) {
    return (int)g[idx + 1];
}

static inline void set_piece(float* g, int idx, int color) {
    g[idx + 1] = (float)color;
}

static inline int get_player(const float* g) {
    return (int)g[0];
}

static inline void set_player(float* g, int p) {
    g[0] = (float)p;
}

// ---------- 围死判定 ----------
static int is_dead(const float* g, int idx, int color) {
    int has_empty = 0, has_friend = 0;
    for (int j = 0; j < BOARD_SIZE; j++) {
        if (distance[idx][j] == 1) {
            int p = (int)g[j+1];
            if (p == 0) has_empty = 1;
            else if (p == color) has_friend = 1;
        }
    }
    return (!has_empty && !has_friend);
}

// ---------- 移位逻辑 ----------
static void shiftOutChessman(float* g) {
    int dead[BOARD_SIZE] = {0};
    for (int i = 0; i < BOARD_SIZE; i++) {
        int color = (int)g[i+1];
        if (color != 0 && is_dead(g, i, color)) {
            dead[i] = 1;
        }
    }
    for (int i = 0; i < BOARD_SIZE; i++) {
        if (dead[i]) g[i+1] = 0.0f;
    }
}

// ---------- 接口实现 ----------

const int numActions(void) {
    init_actions();
    printf("numActions = %d\n", NUM_ACTIONS);  // 加这行
    return NUM_ACTIONS;
}

const int gameLength(void) {
    printf("gameLength = %d\n", BOARD_SIZE + 1);
    return BOARD_SIZE + 1;   // 玩家 + 棋盘
}

const int inputLength(void) {
    // 神经网络输入长度：棋盘 21 个点 + 当前玩家（1维）
    // 可根据你的实际网络输入修改
    return BOARD_SIZE + 1;
}

void rootState(float* const g) {
    // 初始化棋盘：黑棋占据 0,1,2,3,4,8；白棋占据 7,11,12,13,14,15
    for (int i = 0; i < BOARD_SIZE + 1; i++) g[i] = 0.0f;
    g[0] = 1.0f; // 黑先
    int black_init[] = {0,1,2,3,4,8};
    int white_init[] = {7,11,12,13,14,15};
    for (int i = 0; i < 6; i++) {
        g[black_init[i] + 1] = (float)BLACK;
        g[white_init[i] + 1] = (float)WHITE;
    }
}

float playerId(const float* const g) {
    return (float)get_player(g);
}

void inputNetwork(float* const x, const float* const g) {
    // 将状态转换为网络输入：棋盘点（21个） + 当前玩家（1个）
    // 此处简单复制，不进行归一化
    for (int i = 0; i < BOARD_SIZE + 1; i++) {
        x[i] = g[i];
    }
}

int gameEnded(float* const terminal_score, const float* const g) {
    int black = 0, white = 0;
    for (int i = 0; i < BOARD_SIZE; i++) {
        int p = get_piece(g, i);
        if (p == BLACK) black++;
        else if (p == WHITE) white++;
    }
    if (black < 3) {
        *terminal_score = 1.0f;   // 白胜（玩家1胜）
        return 1;
    }
    if (white < 3) {
        *terminal_score = -1.0f;  // 黑胜
        return 1;
    }
    *terminal_score = 0.0f;
    return 0;
}

int isValidAction(const float* const g, int const a) {
    init_actions();
    if (a < 0 || a >= NUM_ACTIONS) return 0;
    int from = ACTION_FROM[a];
    int to = ACTION_TO[a];
    int player = get_player(g);
    if (get_piece(g, from) != player) return 0;
    if (get_piece(g, to) != 0) return 0;
    if (distance[from][to] != 1) return 0;
    return 1;
}

int getValidActions(int* const actions, const float* const g) {
    init_actions();
//    int player = get_player(g);
    int count = 0;
    for (int a = 0; a < NUM_ACTIONS; a++) {
        if (isValidAction(g, a)) {
            actions[count++] = a;
        }
    }
    printf("count = %d\n", count);
    return count;
}

int nextState(float* const ga, const float* const g, const int a) {
    if (!isValidAction(g, a)) return -1;
    // 复制状态
    for (int i = 0; i < BOARD_SIZE + 1; i++) ga[i] = g[i];
    int from = ACTION_FROM[a];
    int to = ACTION_TO[a];
    int player = get_player(ga);
    // 移动
    set_piece(ga, from, 0);
    set_piece(ga, to, player);
    // 移位（移除被围死的棋子）
    shiftOutChessman(ga);
    // 切换玩家
    set_player(ga, -player);
    // 判断是否终局
    float score;
    return gameEnded(&score, ga);
}

void printGame(const float* const g) {
    printf("Current player: %d\n", get_player(g));
    printf("Board: ");
    for (int i = 0; i < BOARD_SIZE; i++) {
        printf("%d ", get_piece(g, i));
    }
    printf("\n");
}