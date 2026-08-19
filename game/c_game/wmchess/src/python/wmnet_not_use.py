import torch
import torch.nn as nn
import torch.nn.functional as F

class WatermelonNet(nn.Module):
    def __init__(self, input_dim=21, num_actions=72, hidden_dim=256):
        """
        input_dim 现在固定为 21（只接收棋盘部分）
        注意：外部调用仍可能传入 22 维的状态，但网络内部会提取 player 并做归一化，
        然后只使用归一化后的棋盘（21维）。
        """
        super().__init__()
        # 第一层输入维度为 21
        self.fc1 = nn.Linear(21, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)

        self.policy_head = nn.Linear(hidden_dim, num_actions)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # x: (batch, 22) 原始状态 [player, board_0, ..., board_20]
        # 提取玩家和棋盘
        player = x[:, 0:1]   # (batch, 1)
        board = x[:, 1:]     # (batch, 21)

        # 视角归一化：当前玩家棋子变为 +1，对手变为 -1
        norm_board = board * player   # (batch, 21)

        # 直接使用归一化棋盘作为输入，不再拼接 player
        x = norm_board  # (batch, 21)

        # 全连接层
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))

        policy_logits = self.policy_head(x)
        value = torch.tanh(self.value_head(x))
        return policy_logits, value