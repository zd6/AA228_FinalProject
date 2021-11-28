from gridDelivery import GridDelivery
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
    plt.hist(np.sum(rewards_stat[:,:,1], axis=1))
    plt.savefig("results/RandomTest.png")

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
    plt.hist(np.sum(rewards_stat[:,:,1], axis=1))
    plt.savefig("results/VITest.png")


if __name__ == "__main__":
    testRandom(time = 1)
    testVI(time = 1)

    

