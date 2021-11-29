import sys
import pdb
  
# setting path
sys.path.append('..')
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
        nn.init.xavier_uniform_(self.l1.weight)
        self.l2= nn.Linear(32, 128)
        nn.init.xavier_uniform_(self.l2.weight)
        self.l3= nn.Linear(128, 64)
        nn.init.xavier_uniform_(self.l2.weight)
        self.head = nn.Linear(64, 4)
    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        return self.head(x)