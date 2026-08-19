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
        self.register_buffer('adj', ADJ_norm)
        # 输入维度改为 1（只有归一化棋盘）
        self.gc1 = GraphConvLayer(1, hidden_dim)
        self.gc2 = GraphConvLayer(hidden_dim, hidden_dim)
        self.gc3 = GraphConvLayer(hidden_dim, hidden_dim)
        self.policy_fc = nn.Linear(hidden_dim, num_actions)
        self.value_fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.from_numpy(x).float().to(self.adj.device)
        else:
            x = x.to(self.adj.device)

        player = x[:, 0:1]
        board = x[:, 1:]
        norm_board = board * player  # (batch, 21)

        # 节点特征：每个节点 1 维（归一化棋子状态）
        node_features = norm_board.unsqueeze(-1)  # (batch, 21, 1)

        h = self.gc1(node_features, self.adj)
        h = self.gc2(h, self.adj)
        h = self.gc3(h, self.adj)

        global_feat = h.mean(dim=1)
        policy_logits = self.policy_fc(global_feat)
        value = torch.tanh(self.value_fc(global_feat))
        return policy_logits, value