from torch.autograd import grad
from gridDelivery import GridDeliveryDQN

import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from DQN.DQN import *

BATCH_SIZE = 1440
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 10

steps_done = 0

env = GridDeliveryDQN()

policy_net = DQN(env.m*env.n)
target_net = DQN(env.m*env.n)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()
optimizer = optim.Adam(policy_net.parameters())
memory = ReplayMemory(10000)


episode_accum_reward = []


def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        np.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randrange(4)]], dtype=torch.long)


def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0]
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    criterion = nn.MSELoss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    grad_list = []
    for param in policy_net.parameters():
        # print("mark", param.grad)
        param.grad.data.clamp_(-1, 1)
        grad_list.append(param.grad.data)
    # print(grad_list)
    # print([torch.linalg.norm(torch.tensor(grad)) for grad in grad_list])
    optimizer.step()


def train():
    num_episodes = 50
    for i_episode in tqdm(range(num_episodes)):
        # Initialize the environment and state
        env.reset()
        state = torch.from_numpy(np.expand_dims(env.dqnState(env.encode_state()), axis=0)).float()
        accum_reward = 0
        day = 0
        while True:
            # Select and perform an action
            action = select_action(state)
            _, _, next_state, reward = env.step(action = action.item())
            accum_reward += reward
            reward = torch.tensor([reward])
            next_state = torch.tensor(np.expand_dims(next_state, axis = 0)).float()
            done = False
            # Observe new state
            if env.hour > 23.99:
                day += 1
            if day == 7:
                done = True
                next_state = None


            # Store the transition in memory
            memory.push([state, action, next_state, reward])

            # Move to the next state
            state = next_state

            # Perform one step of the optimization (on the policy network)
            optimize_model()
            # print(env.hour, accum_reward, done)
            if done:
                episode_accum_reward.append(accum_reward)
                break
        # Update the target network, copying all weights and biases in DQN
        if i_episode % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())
        np.savetxt("trainHistory.txt", np.array(episode_accum_reward))
    print('Complete')
if __name__ == "__main__":
    train()