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
    # np.savetxt("RandomTest.txt", rewards_stat)
    plt.clf()
    plt.xlabel("Reward During 1 Day")
    plt.ylabel("Trails out of 100")
    plt.hist(np.sum(rewards_stat[:,:,1], axis=1))
    plt.savefig("results/RandomTest.png")

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
    plt.clf()
    plt.xlabel("Reward During 1 Day")
    plt.ylabel("Trails out of 100")
    plt.hist(np.sum(rewards_stat[:,:,1], axis=1))
    plt.savefig("results/GreedyTest.png")

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
    plt.clf()
    plt.xlabel("Reward During 1 Day")
    plt.ylabel("Trails out of 100")
    plt.hist(np.sum(rewards_stat[:,:,1], axis=1))
    plt.savefig("results/VITest.png")

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
    plt.clf()
    plt.xlabel("Reward During 1 Day")
    plt.ylabel("Trails out of 100")
    plt.hist(np.sum(rewards_stat[:,:,1], axis=1))
    plt.savefig("results/DQNTest.png")

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
    plt.clf()
    plt.hist(np.sum(rewards_stat[:,:,1], axis=1))
    plt.xlabel("Reward During 1 Day")
    plt.ylabel("Trails out of 100")
    plt.savefig("results/SampleVITest.png")



if __name__ == "__main__":
    # testRandom(time = 1)
    testGreedy(time = 1)
    # testVI(time = 1)
    # testSampleVI(time = 1)
    # testDQN(time = 1)

    

