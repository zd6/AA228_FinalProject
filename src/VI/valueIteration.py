import numpy as np
from gridDelivery import directions
# U[s], T[s,a,s'], R[s,a]



def VI(grid):
    U = np.zeros(grid.m*grid.n)
    T = np.zeros((grid.m*grid.n, 4, grid.m*grid.n))
    for i in range(grid.m):
        for j in range(grid.n):
            idx = pos_2_idx(i, j, grid)
            for a, (x, y) in enumerate(directions):
                if i+x > 0 and j+y > 0 and i+x < grid.m and j+y < grid.n:
                    newIdx = pos_2_idx(i+x, j+y, grid)
                    T[idx, a, newIdx] =

def pos_2_idx(x, y, grid):
    return x*grid.m + y
def idx_2_pos(idx, grid):
    return idx//grid.m, idx%grid.n


def valueIteration(U, T, R, gamma, stopping = 1e-5, max_k = 10000):
    k = 0
    while k < max_k:
        newU = np.max(R + gamma*np.tensordot(T, U, (2, 0)), axis = 1)
        if np.linalg.norm(newU - U) < stopping:
            return U
        U = newU
        k += 1
    return U