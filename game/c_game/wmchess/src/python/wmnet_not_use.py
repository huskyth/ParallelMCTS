import torch
import torch.nn as nn
import torch.nn.functional as F

class WatermelonNet(nn.Module):
    def __init__(self, input_dim, num_actions, hidden_dim=256):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)

        # 策略头 (Policy Head)
        self.policy_head = nn.Linear(hidden_dim, num_actions)

        # 价值头 (Value Head)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))

        # 策略输出（需要 softmax，但在 loss 中可以用 CrossEntropy，所以这里保留 logits）
        policy_logits = self.policy_head(x)
        # 价值输出（tanh 限制在 -1 到 1 之间）
        value = torch.tanh(self.value_head(x))
        return policy_logits, value