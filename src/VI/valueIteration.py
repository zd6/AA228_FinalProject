import numpy as np
from const import *
# U[s], T[s,a,s'], R[s,a]

class VI:
    def __init__(self, grid):
        self.grid = grid
        self.replan()
    def replan(self):
        self.costKey = "DAY" if self.grid.is_rush() else "NIGHT"
        T = np.zeros((self.grid.m*self.grid.n + 1, 4, self.grid.m*self.grid.n + 1))
        T[-1, :, -1] = [1,1,1,1]
        for i in range(self.grid.m):
            for j in range(self.grid.n):
                idx = self.pos_2_idx(i, j)
                if self.grid.grid[PACKAGE][i, j] == 1:
                    T[idx, :, -1] = [1,1,1,1]
                else:
                    stuck_prob = self.stuck_prob(i, j)
                    for a, (x, y) in enumerate(directions):
                        if self.in_bound(i, j, x, y):
                            newIdx = self.pos_2_idx(x+i, y+j)
                            T[idx, a, newIdx] = 1 - stuck_prob/100
                            T[idx, a, idx] = stuck_prob/100
                        else:
                            T[idx, a, idx] = 1
        self.T = T
        U = np.zeros(self.grid.m*self.grid.n + 1)
        self.R = np.ones((self.grid.m*self.grid.n+1, 4))*self.grid.rewards[self.costKey]
        self.R[-1,:] = [0,0,0,0]
        for x, y in self.grid.packages_pos_to_id.keys():
            self.R[self.pos_2_idx(x, y),:] += self.grid.rewards["PACKAGE"]
        self.U = self.valueIteration(U,self.T,self.R,0.99)
        self.curPkg = list(self.grid.packages.values())
    
    def policy(self, state):
        _, pos, pkgs = state
        if self.curPkg != pkgs:
            self.curPkg = pkgs
            self.replan()
        cur = self.pos_2_idx(*pos)
        actionReward = self.R[cur, :] + np.tensordot(self.T[cur], self.U, axes=[1, 0])
        return np.argmax(actionReward)

    def stuck_prob(self, i, j):
        if self.grid.is_highway(i, j):
            return self.grid.traffic_prob["HIGHWAY"][self.costKey == "NIGHT"]
        else:
            return self.grid.traffic_prob["STREET"][self.costKey == "NIGHT"]
    def in_bound(self, i, j, x, y):
        return i+x >= 0 and j+y >= 0 and i+x < self.grid.m and j+y < self.grid.n
    def pos_2_idx(self, x, y):
        return x*self.grid.m + y
    def idx_2_pos(self, idx):
        return idx//self.grid.m, idx%self.grid.n


    def valueIteration(self, U, T, R, gamma, stopping = 1e-5, max_k = 10000):
        k = 0
        while k < max_k:
            newU = np.max(R + gamma*np.tensordot(T, U, axes=[2, 0]), axis = 1)
            if np.linalg.norm(newU - U) < stopping:
                return U
            U = newU
            k += 1
        return U