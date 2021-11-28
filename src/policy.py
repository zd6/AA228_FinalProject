from random import randint
import numpy as np
from VI.valueIteration import VI

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
# U[s], T[s,a,s'], R[s,a]
