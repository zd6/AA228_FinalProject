from gridDelivery import GridDeliveryDQN

from collections import deque, namedtuple
import random

import torch.nn as nn
import torch.nn.functional as F

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class ReplayMemory:
    def __init__(self, cap) -> None:
        self.memory = deque([], maxlen = cap)
    def push(self, sasr):
        self.memory.append(Transition(*sasr))
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    def __len__(self):
        return len(self.memory)

OUTPUT = 4
class DQN(nn.Module):
    def __init__(self, inputSize) -> None:
        super(DQN, self).__init__()
        self.l1 = nn.Linear(inputSize, 32)
        self.l2= nn.Linear(32, 128)
        self.l3 = nn.Linear(128, 256)
        self.head = nn.Linear(256, 4)
    def forward(self, x):
        x = F.normalize(x)
        x = F.relu(F.normalize((self.l1(x))))
        x = F.relu(F.normalize(self.l2(x)))
        x = F.relu(F.normalize(self.l3(x)))
        return self.head(x)