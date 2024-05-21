import random
import numpy as np
import torch
import torch.nn as nn
from itertools import product

import torch.nn.functional as F
import pickle
from torch.distributions import Categorical
from PointRobotEnv import PointEnv_MultiStep_Two_goal
use_cuda = torch.cuda.is_available()
device   = torch.device("cuda" if use_cuda else "cpu")
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

class MAReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, obs, action, reward, next_obs, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (obs, action, reward, next_obs, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        obs, action, reward, next_obs, done = map(np.stack, zip(*batch))
        return obs, action, reward, next_obs, done

    def __len__(self):
        return len(self.buffer)

def save_variable(v,filename):
    f = open(filename, 'wb')
    pickle.dump(v, f)
    f.close()
    return filename

def load_variavle(filename):
    f = open(filename, 'rb')
    r = pickle.load(f)
    f.close()
    return r

class Retrieval(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super(Retrieval, self).__init__()

        self.l1 = nn.Linear(obs_dim + action_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 256)
        self.l4 = nn.Linear(256, obs_dim)

        self.obs_dim = obs_dim
        self.action_dim = action_dim

    def forward(self, observations, actions):
        oa = torch.cat([observations, actions], -1)

        q1 = F.relu(self.l1(oa))
        q1 = F.relu(self.l2(q1))
        q1 = F.relu(self.l3(q1))
        q1 = self.l4(q1)

        return q1

class Network(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim):
        super(Network, self).__init__()

        self.l1 = nn.Linear(obs_dim + action_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, 1)

    def forward(self, obs, action):
        oa = torch.cat([obs, action], -1)

        q1 = F.relu(self.l1(oa))
        q1 = F.relu(self.l2(q1))
        q1 = F.softplus(self.l3(q1))
        return q1

class FN(object):
    def __init__(
            self,
            obs_dim,
            action_dim,
            hidden_dim,
            max_action,
            discount=0.99,
            tau=0.005,
            policy_noise=0.2,
            noise_clip=0.5,
            policy_freq=2
    ):

        self.network = Network(obs_dim, action_dim, hidden_dim).to(device)
        self.network_optimizer = torch.optim.Adam(self.network.parameters(), lr=3e-4)
        self.retrieval = Retrieval(obs_dim, action_dim).to(device)
        self.retrieval_optimizer = torch.optim.Adam(self.retrieval.parameters(), lr=3e-5)

        self.max_action = max_action
        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq

    def select_action(self, obs, is_max):
        with torch.no_grad():
            obs = torch.FloatTensor(obs.reshape(1, -1)).repeat(self.max_action, 1).to(device)
            sample_action = np.arange(self.max_action)
            edge_flow = self.network(obs, sample_action).reshape(-1)
            if is_max == 0:
                idx = Categorical(edge_flow.float()).sample(torch.Size([1]))
                action = sample_action[idx[0]]
            elif is_max == 1:
                action = sample_action[edge_flow.argmax()]
        return action.cpu().data.numpy().flatten()

    def train(self, replay_buffer, frame_idx, batch_size=256, max_episode_steps=50, sample_flow_num=100):
        # Sample replay buffer
        state, action, reward, next_state, not_done = replay_buffer.sample(batch_size)
        state = torch.FloatTensor(state).to(device)
        next_state = torch.FloatTensor(next_state).to(device)
        action = torch.FloatTensor(action).to(device)
        reward = torch.FloatTensor(reward).to(device)
        not_done = torch.FloatTensor(np.float32(not_done)).to(device)

        with torch.no_grad():
            

            uniform_action = np.random.uniform(low=-max_action, high=max_action,
                                               size=(batch_size, max_episode_steps, sample_flow_num, action_dim))
            uniform_action = torch.Tensor(uniform_action).to(device)
            current_state = next_state.repeat(1, 1, sample_flow_num).reshape(batch_size, max_episode_steps,
                                                                             sample_flow_num, -1)
            inflow_state = self.retrieval(current_state, uniform_action)
            inflow_state = torch.cat([inflow_state, state.reshape(batch_size, max_episode_steps, -1, state_dim)], -2)
            uniform_action = torch.cat([uniform_action, action.reshape(batch_size, max_episode_steps, -1, action_dim)], -2)
        edge_inflow = self.network(inflow_state, uniform_action).reshape(batch_size, max_episode_steps, -1)
        epi = torch.Tensor([1.0]).repeat(batch_size*max_episode_steps).reshape(batch_size,-1).to(device)
        inflow = torch.log(torch.sum(torch.exp(torch.log(edge_inflow)), -1) + epi)

        with torch.no_grad():
            uniform_action = np.random.uniform(low=-max_action, high=max_action,
                                               size=(batch_size, max_episode_steps, sample_flow_num, action_dim))
            uniform_action = torch.Tensor(uniform_action).to(device)
            outflow_state = next_state.repeat(1, 1, (sample_flow_num+1)).reshape(batch_size, max_episode_steps, (sample_flow_num+1), -1)
            last_action = torch.Tensor([0.0]).reshape([1,1,1]).repeat(batch_size,1,1).to(device)
            last_action = torch.cat([action[:,1:,:], last_action], -2)
            uniform_action = torch.cat([uniform_action, last_action.reshape(batch_size, max_episode_steps, -1, action_dim)], -2)

        edge_outflow = self.network(outflow_state, uniform_action).reshape(batch_size, max_episode_steps, -1)

        outflow = torch.log(torch.sum(torch.exp(torch.log(edge_outflow)), -1) + epi)

        network_loss = F.mse_loss(inflow * not_done, outflow * not_done, reduction='none') + F.mse_loss(inflow * done_true, (torch.cat([reward[:,:-1],torch.log((reward*(sample_flow_num+1))[:,-1]).reshape(batch_size,-1)], -1)) * done_true, reduction='none')
        network_loss = torch.mean(torch.sum(network_loss, dim = 1))
        print(network_loss)
        self.network_optimizer.zero_grad()
        network_loss.backward()
        self.network_optimizer.step()

        if frame_idx % 5 == 0:
            pre_state = self.retrieval(next_state, action)
            retrieval_loss = F.mse_loss(pre_state, state)
            print(retrieval_loss)

            # Optimize the network
            self.retrieval_optimizer.zero_grad()
            retrieval_loss.backward()
            self.retrieval_optimizer.step()