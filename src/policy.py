from random import randint
import numpy as np
import torch
from VI.valueIteration import VI, SampleVI
from DQN.model import DQN

CHANNELS = 3
ROAD = 0
CITY = 1
PACKAGE = 2
DAY = 0
NIGHT = 1
PACKAGE_APPR = 10
directions = [[-1,0],[1,0],[0,-1],[0,1]]

class RandomPolicy:
    def __init__(self, env) -> None:
        pass
    def policy(self, state):
        return randint(0,3)
class GreedyPolicy:
    def __init__(self, env) -> None:
        pass
    def policy(self, state):
        truck = state[1]
        pkgs = state[-1]
        dist = [(pkg[0]-truck[0])**2 + (pkg[1]-truck[1])**2 for pkg in pkgs]
        nearest = state[-1][np.argmin(dist)]
        # print(state, nearest, np.argmax(np.dot(directions, [nearest[0] - truck[0], nearest[1] - truck[1]])))
        return np.argmax(np.dot(directions, [nearest[0] - truck[0], nearest[1] - truck[1]]))
class DQNPolicy:
    def __init__(self, env) -> None:
        self.policy_net = DQN(9)
        self.policy_net.load_state_dict(torch.load("DQN/nets/policy_net_99.pickle"))
    def policy(self, state):
        return self.policy_net(torch.tensor(state).float()).max(0)[1].item()
# U[s], T[s,a,s'], R[s,a]
