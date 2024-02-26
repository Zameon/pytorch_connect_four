from pettingzoo.classic import connect_four_v3
import gymnasium as gym
import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import time

is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display


env = connect_four_v3.env()


plt.ion()

# if GPU is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
    
class DQN(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)
    

# BATCH_SIZE is the number of transitions sampled from the replay buffer
# GAMMA is the discount factor as mentioned in the previous section
# EPS_START is the starting value of epsilon
# EPS_END is the final value of epsilon
# EPS_DECAY controls the rate of exponential decay of epsilon, higher means a slower decay
# TAU is the update rate of the target network
# LR is the learning rate of the ``AdamW`` optimizer
BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.1
EPS_DECAY = 100
TAU = 0.005
LR = 1e-4

# Get number of actions from gym action space
# Get the number of state observations
env.reset()
n_actions = env.action_space(env.agents[0]).n
var = env.observation_space(env.agents[0])['observation'].shape
n_observations = var[0]*var[1]*var[2]


policy_net = DQN(n_observations, n_actions).to(device)
target_net = DQN(n_observations, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
memory = ReplayMemory(10000)

steps_done = 0


def select_action(state, mask, agent, eps_threshold):
    global steps_done
   
    sample = random.random()
  
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return the largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            maxidx = 0
            temp = -float('inf')
            maxi = policy_net(state)
            for i in range(len(maxi[0])):
                if mask[i] == True and maxi[0][i] >= temp:
                    maxidx = i
                    temp = maxi[0][maxidx]

              
            return torch.tensor(maxidx).view(1, 1)
    else:
        return torch.tensor([[env.action_space(agent).sample(mask)]], device=device, dtype=torch.long)


env = connect_four_v3.env(render_mode="human")
env.reset(seed=42)


player0 = 0
player1 = 0
policy_net = torch.load("connect_four.pt")


for test_epoch in range(3):
    print(f"Epoch number: {test_epoch}")
    env.reset()

    for test_agent in env.agent_iter():

        observation, reward, termination, truncation, info = env.last()
        #print(f"Rewards: {env.rewards}")
    
        if termination or truncation:
            
            action = None
            print(f"Rewards: {env.rewards}")
            if env.rewards['player_0'] >=0:
                player0 += env.rewards['player_0']
            if env.rewards['player_1'] >=0:
                player1 += env.rewards['player_1']
            break
        else:
            test_mask = observation["action_mask"]
            test_observation = torch.tensor(observation['observation'], device=device, dtype=torch.float32).flatten().unsqueeze(0)
            if test_agent == 'player_0':
                action = select_action(test_observation, test_mask, test_agent, 0.01 )  # this is where you would insert your policy
            else:
                action = select_action(test_observation, test_mask, test_agent, 1.0 )     
        env.step(action.item())
        time.sleep(1.5)


print(f"player 0(Red) : {player0}")
print(f"player 1(Black) : {player1}")


