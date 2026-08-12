import random
from collections import deque
import numpy as np

class ReplayBuffer:
    def __init__(self, max_size=100000):
        self.buffer = deque(maxlen=max_size)

    def push(self, state, policy, z):
        self.buffer.append((state.copy(), policy.copy(), z))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, policies, zs = zip(*batch)
        return np.array(states, dtype=np.float32), np.array(policies, dtype=np.float32), np.array(zs, dtype=np.float32).reshape(-1, 1)

    def __len__(self):
        return len(self.buffer)