from gridDelivery import GridDelivery, GridDeliveryDQN
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
def testRandom(time = 5):
    env = GridDelivery()
    env.config(configFile="config_random.json")
    rewards_stat = []
    for _ in tqdm(range(100)):
        rewards = []
        for _ in range(time*24*60):
            _,_,reward,_ = env.step()
            rewards.append([env.is_rush(), reward])
        rewards_stat.append(rewards)
    rewards_stat = np.array(rewards_stat)
    np.save("results/RandomTest", rewards_stat)

def testGreedy(time = 5):
    env = GridDelivery()
    env.config(configFile="config_greedy.json")
    rewards_stat = []
    for _ in tqdm(range(100)):
        rewards = []
        for _ in range(time*24*60):
            _,_,reward,_ = env.step()
            rewards.append([env.is_rush(), reward])
        rewards_stat.append(rewards)
    rewards_stat = np.array(rewards_stat)
    # np.savetxt("RandomTest.txt", rewards_stat)
    np.save("results/GreedyTest", rewards_stat)


def testVI(time = 1):
    env = GridDelivery()
    env.config(configFile="config_VI.json")
    rewards_stat = []
    for _ in tqdm(range(100)):
        rewards = []
        for _ in range(time*24*60):
            _,_,reward,_ = env.step()
            rewards.append([env.is_rush(), reward])
        rewards_stat.append(rewards)
    rewards_stat = np.array(rewards_stat)
    # np.savetxt("RandomTest.txt", rewards_stat)
    np.save("results/VITest", rewards_stat)


def testDQN(time = 1):
    rewards_stat = []
    for _ in tqdm(range(100)):
        rewards = []
        env = GridDeliveryDQN()
        for _ in range(time*24*60):
            _,_,reward,_ = env.step()
            rewards.append([env.is_rush(), reward])
        rewards_stat.append(rewards)
    rewards_stat = np.array(rewards_stat)
    # np.savetxt("RandomTest.txt", rewards_stat)
    np.save("results/DQNTest", rewards_stat)


def testSampleVI(time = 1):
    env = GridDelivery()
    env.config(configFile="config_SampleVI.json")
    rewards_stat = []
    for _ in tqdm(range(100)):
        rewards = []
        for _ in range(time*24*60):
            _,_,reward,_ = env.step()
            rewards.append([env.is_rush(), reward])
        rewards_stat.append(rewards)
    rewards_stat = np.array(rewards_stat)
    # np.savetxt("RandomTest.txt", rewards_stat)
    np.save("results/SampleVITest", rewards_stat)


import time
if __name__ == "__main__":
    start = time.time()
    testRandom(time = 1)
    print("Random Policy Runtime", time.time() - start)
    start = time.time()
    testGreedy(time = 1)
    print("Greedy Policy Runtime", time.time() - start)
    start = time.time()
    testVI(time = 1)
    print("VI Policy Runtime", time.time() - start)
    start = time.time()
    testSampleVI(time = 1)
    print("SampleVI Policy Runtime", time.time() - start)
    start = time.time()
    testDQN(time = 1)
    print("DQN Policy Runtime", time.time() - start)

    

