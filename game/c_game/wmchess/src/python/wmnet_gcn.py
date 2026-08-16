# wmnet_gcn.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# 21x21 邻接矩阵（从距离矩阵生成，距离为1则为边）
# 这里直接硬编码（你之前提供的距离矩阵）
ADJ = np.array([
    [0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    [1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1],
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0]
], dtype=np.float32)

# 添加自环
ADJ = ADJ + np.eye(21)

# 归一化邻接矩阵（对称归一化）
D = np.sum(ADJ, axis=1)
D_inv_sqrt = np.diag(1.0 / np.sqrt(D))
ADJ_norm = D_inv_sqrt @ ADJ @ D_inv_sqrt
ADJ_norm = torch.tensor(ADJ_norm, dtype=torch.float32)


class GraphConvLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x, adj):
        # x: (batch, nodes, in_features)
        # adj: (nodes, nodes)
        x = self.linear(x)  # (batch, nodes, out)
        x = torch.bmm(adj.unsqueeze(0).expand(x.size(0), -1, -1), x)  # (batch, nodes, out)
        return F.relu(x)


class WatermelonGCN(nn.Module):
    def __init__(self, num_actions=72, hidden_dim=128):
        super().__init__()
        self.adj = ADJ_norm  # (21, 21)

        # 特征维度：棋子颜色（1维）+ 当前玩家（1维） = 2
        self.gc1 = GraphConvLayer(2, hidden_dim)
        self.gc2 = GraphConvLayer(hidden_dim, hidden_dim)
        self.gc3 = GraphConvLayer(hidden_dim, hidden_dim)

        # 全局池化后接全连接
        self.policy_fc = nn.Linear(hidden_dim, num_actions)
        self.value_fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        """
        x: (batch, 22) 原始状态 [player, board_0, ..., board_20]
        其中 board_i 为 1, -1, 0
        """
        # 分离玩家和棋盘
        player = x[:, 0:1]  # (batch, 1)
        board = x[:, 1:]  # (batch, 21)

        # 构造节点特征：每个节点拼接 [棋子颜色, 当前玩家]
        # 将玩家信息广播到每个节点
        node_features = torch.stack([board, player.expand(-1, 21)], dim=-1)  # (batch, 21, 2)

        # GCN 前向
        h = self.gc1(node_features, self.adj.to(node_features.device))
        h = self.gc2(h, self.adj.to(h.device))
        h = self.gc3(h, self.adj.to(h.device))

        # 全局平均池化（所有节点取平均）
        global_feat = h.mean(dim=1)  # (batch, hidden_dim)

        # 策略头
        policy_logits = self.policy_fc(global_feat)  # (batch, num_actions)
        # 价值头
        value = torch.tanh(self.value_fc(global_feat))  # (batch, 1)

        return policy_logits, value