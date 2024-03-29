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

# Need this in order to learn and store the experiences.
# Contains a deque to store the "Transition tuples" which are the experiences. 
# Sampling returns a random number(arg batch_size) of transitions as a mini-batch that is required for stuff 

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
        self.layer1 = nn.Linear(n_observations, 128) # 1st Hidden layer
        self.layer2 = nn.Linear(128, 128) # 2nd Hidden layer
        self.layer3 = nn.Linear(128, n_actions) # Output Layer 

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.layer1(x)) #relu activation function on the first layer 
        x = F.relu(self.layer2(x)) #relu activation function on the second layer
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
EPS_START = 1
EPS_END = 0.1
EPS_DECAY = 10000
TAU = 0.005
LR = 1e-4
# Get number of actions from gym action space
# Get the number of state observations
env.reset()
n_actions = env.action_space(env.agents[0]).n # from the documentation 

var = env.observation_space(env.agents[0])['observation'].shape # from the documentation 
n_observations = var[0]*var[1]*var[2]


policy_net = DQN(n_observations, n_actions).to(device)
target_net = DQN(n_observations, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True) # don't need to know 
memory = ReplayMemory(10000) #replay buffer that stores the experiences or transition tuples 

steps_done = 0

# selects action randomly or based on past experiences. That probability is chosen using the EPS. 
# at the beginning of learning, all the actions are randomised. this "random" value decays (EPS_DECAY) over time 

def select_action(state, mask, agent):
    global steps_done
   
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
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

              
            return torch.tensor(maxidx).view(1, 1) # returns the index of the action that has the maximised value of q 
    else:
        return torch.tensor([[env.action_space(agent).sample(mask)]], device=device, dtype=torch.long) # returns a random action


episode_durations = []


def plot_durations(show_result=False):
    plt.figure(1)
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    if show_result:
        plt.title('Result')
    else:
        plt.clf()
        plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        if not show_result:
            display.display(plt.gcf())
            display.clear_output(wait=True)
        else:
            display.display(plt.gcf())
def optimize_model():
    if len(memory) < BATCH_SIZE: # if enough experiences have not been gathered yet, a mini batch can't be performed 
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool,)
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
    # on the "older" target_net; selecting their best reward with max(1).values
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1).values
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch # BELLMAN EQN 

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))
   
    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()

if torch.cuda.is_available():
    num_episodes = 10000
else:
    num_episodes = 10000

# MAIN LOOP 
    
for i_episode in range(num_episodes):
    # Initialize the environment and get it's state
    env.reset()
    observation, reward, termination, truncation, info = env.last()
    observation_hat = torch.tensor(observation['observation'], device=device, dtype=torch.float32).flatten().unsqueeze(0) # basically flattened thing 
    t=0

    for agent in env.agent_iter():
        mask = observation["action_mask"] # does this to choose legal moves, acc. to documentation 
        action = select_action(observation_hat, mask, agent)
        env.step(action.item())
        observation, reward, termination, truncation, info = env.last()
        t += 1
        if termination:
            new_observation_hat = None
            
        else:
            
            new_observation_hat = torch.tensor(observation['observation'], device=device, dtype=torch.float32).flatten().unsqueeze(0)

        reward = torch.tensor([reward], device=device)
        memory.push(observation_hat, action, new_observation_hat, reward)
        observation_hat = new_observation_hat
        optimize_model()

        target_net_state_dict = target_net.state_dict()
        policy_net_state_dict = policy_net.state_dict()

        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)

        target_net.load_state_dict(target_net_state_dict)

        if termination or truncation:
            episode_durations.append(t)
            plot_durations()
            break


print('Complete')
torch.save(policy_net, "connect_four.pt")
plot_durations(show_result=True)
plt.ioff()
plt.show()




