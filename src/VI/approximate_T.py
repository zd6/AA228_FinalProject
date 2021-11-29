from random import sample
import sys
from tqdm import tqdm
# setting path
sys.path.append('..')
import numpy as np
  

def sampleEnv(env):
    samples = []
    for i in tqdm(range(1000000)):
        prevState, action, _, nextState = env.step()
        samples.append([prevState[0], *prevState[1], action, *nextState[1]])
    np.savetxt("samples.txt", np.array(samples, dtype=int))

def constructT(env):
    print('Reconstructing T')
    stat = np.loadtxt("VI/samples.txt")
    not_rush_samples = stat[np.where(stat[:,0] == 0)].astype(int)
    rush_samples = stat[np.where(stat[:,0] == 1)].astype(int)
    rush_T = np.zeros((env.m*env.n+1, 4, env.m*env.n+1))
    rush_N = np.zeros((env.m*env.n+1, 4))
    rush_T = count(rush_samples, rush_T, rush_N)
    not_rush_T = np.zeros((env.m*env.n+1, 4, env.m*env.n+1))
    not_rush_N = np.zeros((env.m*env.n+1, 4))
    not_rush_T = count(not_rush_samples, not_rush_T, not_rush_N)
    return rush_T, not_rush_T
def count(stat, T, N):
    for sample in stat:
        T[sample[1]*5+sample[2], sample[3], sample[4]*5+sample[5]] += 1
        N[sample[1]*5+sample[2], sample[3]] += 1
    T = np.einsum('ijk, ij -> ijk', T, np.divide(1, N, out=np.zeros_like(N), where=N!=0))
    return T
if __name__ == '__main__':
    constructT()