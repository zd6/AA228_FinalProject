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
    env = GridDeliveryDQN()
    env.config(configFile="config_DQN.json")
    rewards_stat = []
    for _ in tqdm(range(100)):
        rewards = []
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



if __name__ == "__main__":
    testRandom(time = 1)
    testGreedy(time = 1)
    testVI(time = 1)
    testSampleVI(time = 1)
    testDQN(time = 1)

    

