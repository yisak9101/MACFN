import random
import numpy as np
import torch
import torch.nn as nn
from pettingzoo.mpe import simple_spread_v3
from pettingzoo.mpe._mpe_utils.simple_env import SimpleEnv

import torch.nn.functional as F
import pickle
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

now_time = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

class RewardShapeWrapper(SimpleEnv):
    def _reward(self, r, done):
        sparse_r = 0
        if done and r >= -0.09:
            sparse_r = r + 0.2
        return sparse_r

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        if done == False:
            reward = 0
        else:
            reward = info['reward_dist']
        return observation, reward, done, info

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


def save_variable(v, filename):
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


class CFN(object):
    def __init__(
            self,
            n_agents,
            obs_dim,
            action_dim,
            hidden_dim,
            min_action,
            max_action,
            uniform_action_size,
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

        self.n_agents = n_agents
        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq
        self.action_dim = action_dim
        self.min_action = min_action
        self.max_action = max_action

        self.uniform_action_size = uniform_action_size
        self.uniform_action = np.random.uniform(low=self.min_action, high=self.max_action,
                                                size=(self.uniform_action_size, self.action_dim))
        self.uniform_action = torch.Tensor(self.uniform_action).to(device)

        self.obs_dim = obs_dim
        self.action_dim = action_dim

    def select_action(self, obs, is_max):
        sample_action = np.random.uniform(low=self.min_action, high=self.max_action, size=(self.uniform_action_size, self.action_dim))
        with torch.no_grad():
            sample_action = torch.Tensor(sample_action).to(device)
            state = torch.FloatTensor(obs.reshape(1, -1)).repeat(self.uniform_action_size, 1).to(device)
            edge_flow = self.network(state, sample_action).reshape(-1)
            if is_max == 0:
                idx = Categorical(edge_flow.float()).sample(torch.Size([1]))
                action = sample_action[idx[0]]
            elif is_max == 1:
                action = sample_action[edge_flow.argmax()]
        return action.cpu().data.numpy().flatten()

    def set_uniform_action(self):
        self.uniform_action = np.random.uniform(low=self.min_action, high=self.max_action,
                                                size=(self.uniform_action_size, self.action_dim))
        self.uniform_action = torch.Tensor(self.uniform_action).to(device)
        return self.uniform_action

    def train(self, replay_buffer, frame_idx, batch_size=256, max_episode_steps=50, sample_flow_num=100):
        # Sample replay buffer
        obs, action, reward, next_obs, not_done = replay_buffer.sample(batch_size)
        obs = torch.FloatTensor(obs).to(device)
        next_obs = torch.FloatTensor(next_obs).to(device)
        action = torch.FloatTensor(action).to(device)
        reward = torch.FloatTensor(reward).to(device)
        not_done = torch.FloatTensor(np.float32(not_done)).to(device)

        with torch.no_grad():
            uniform_action = np.random.uniform(low=self.min_action, high=self.max_action,
                                               size=(batch_size, max_episode_steps, self.n_agents, sample_flow_num,
                                                     self.action_dim))
            uniform_action = torch.Tensor(uniform_action).to(device)
            current_obs = next_obs.repeat(1, 1, 1, sample_flow_num).reshape(batch_size, max_episode_steps,
                                                                            self.n_agents,
                                                                            sample_flow_num, -1)
            inflow_state = self.retrieval(current_obs,
                                          uniform_action)  # (batch_size, max_episode_steps, self.n_agents, sample_flow_num, self.obs_dim)
            inflow_state = torch.cat(
                [inflow_state, obs.reshape(batch_size, max_episode_steps, self.n_agents, -1, self.obs_dim)], -2)
            uniform_action = torch.cat(
                [uniform_action, action.reshape(batch_size, max_episode_steps, self.n_agents, -1, self.action_dim)],
                -2)

        epi = torch.ones((batch_size, max_episode_steps)).to(device)

        edge_inflow = self.network(inflow_state, uniform_action).reshape(batch_size, max_episode_steps, self.n_agents,
                                                                         -1)
        inflow = torch.log(torch.sum(torch.exp(torch.log(edge_inflow)), -1).prod(dim=-1) + epi)  # (batch_size, max_episode_steps)

        with torch.no_grad():
            uniform_action = np.random.uniform(low=self.min_action, high=self.max_action,
                                               size=(batch_size, max_episode_steps, self.n_agents, sample_flow_num,
                                                     self.action_dim))
            uniform_action = torch.Tensor(uniform_action).to(device)
            outflow_obs = next_obs.repeat(1, 1, 1, (sample_flow_num + 1)).reshape(batch_size, max_episode_steps,
                                                                                    self.n_agents,
                                                                                    (sample_flow_num + 1), -1)
            last_action = torch.zeros((batch_size, 1, self.n_agents, self.action_dim)).to(device)
            last_action = torch.cat([action[:, 1:, :, :], last_action],
                                    1)  # (batch_size, max_episode_steps, self.n_agents, self.action_dim)
            uniform_action = torch.cat(
                [uniform_action, last_action.reshape(batch_size, max_episode_steps, self.n_agents, 1, self.action_dim)],
                -2) # (batch_size, max_episode_steps, self.n_agents, sample_flow_num + 1, self.action_dim)

        edge_outflow = self.network(outflow_obs, uniform_action).reshape(batch_size, max_episode_steps, self.n_agents, -1)

        outflow = torch.log(torch.sum(torch.exp(torch.log(edge_outflow)), -1).prod(dim=-1) + reward + epi)

        network_loss = F.mse_loss(inflow, outflow, reduction='none')
        network_loss = torch.mean(torch.sum(network_loss, dim=1))
        print(network_loss)
        self.network_optimizer.zero_grad()
        network_loss.backward()
        self.network_optimizer.step()

        if frame_idx % 5 == 0:
            pre_state = self.retrieval(next_obs, action)
            retrieval_loss = F.mse_loss(pre_state, obs)
            print(retrieval_loss)

            # Optimize the network
            self.retrieval_optimizer.zero_grad()
            retrieval_loss.backward()
            self.retrieval_optimizer.step()

writer = SummaryWriter(log_dir="runs/MACFN_Spread_"+now_time)

max_episode_steps = 25
n_agents = 3

env = simple_spread_v3.parallel_env(N=n_agents, max_cycles=max_episode_steps, continuous_actions=True, render_mode='human')
test_env = simple_spread_v3.parallel_env(N=n_agents, max_cycles=max_episode_steps, continuous_actions=True)

env.reset()
test_env.reset()

obs_dim = 18
action_dim = 5
min_action = 0
max_action = 1
hidden_dim = 256
uniform_action_size = 1000

policy = CFN(n_agents, obs_dim, action_dim, hidden_dim, min_action, max_action, uniform_action_size)
policy.retrieval.load_state_dict(torch.load('retrieval_spread.pkl'))
policy.network.load_state_dict(torch.load('MACFN_spread.pkl'))

replay_buffer_size = 8000
replay_buffer = MAReplayBuffer(replay_buffer_size)

max_frames  = 1666
start_timesteps = 260
frame_idx = 0
episode_rewards = []
test_rewards = []
x_idx = []
batch_size = 200
test_epoch = 0
expl_noise = 0.4
sample_flow_num = 99
repeat_episode_num = 5
sample_episode_num = 1000

while frame_idx < max_frames:
    observations, _ = env.reset()
    episode_reward = 0

    obs_buf = []
    action_buf = []
    reward_buf = []
    next_obs_buf = []
    not_done_buf = []

    for step in range(max_episode_steps):
        with torch.no_grad():
            actions = {agent: policy.select_action(observations[agent], 0) for agent in env.agents}

        next_observations, rewards, dones, truncations, infos = env.step(actions)
        done = (True in dones.values() or True in truncations.values())
        reward = sum(rewards.values()) / len(rewards.values())

        obs_buf.append(np.array(list(observations.values())))
        action_buf.append(np.array(list(actions.values())))
        reward_buf.append(reward)
        next_obs_buf.append(np.array(list(next_observations.values())))
        not_done_buf.append(1 - done)

        observations = next_observations
        episode_reward += reward

        if done:
            frame_idx += 1
            print(frame_idx)
            replay_buffer.push(obs_buf, action_buf, reward_buf, next_obs_buf, not_done_buf)
            break

        if frame_idx >= start_timesteps and step % 2 == 0:
            policy.train(replay_buffer, frame_idx, batch_size, max_episode_steps, sample_flow_num)

    episode_rewards.append(episode_reward)

    if frame_idx > start_timesteps and frame_idx % 25 == 0:
        print(frame_idx)
        test_epoch += 1
        avg_test_episode_reward = 0
        for i in range(repeat_episode_num):
            test_observations, _ = test_env.reset()
            test_episode_reward = 0
            for s in range(max_episode_steps):
                with torch.no_grad():
                    test_actions = {agent: policy.select_action(test_observations[agent], 1) for agent in test_env.agents}

                test_next_observations, test_rewards, test_dones, test_truncations, test_infos = test_env.step(test_actions)
                test_done = (True in test_dones.values() or True in test_truncations.values())
                test_reward = sum(test_rewards.values()) / len(test_rewards.values())
                test_observations = test_next_observations
                test_episode_reward += test_reward
                if test_done:
                    break
            avg_test_episode_reward += test_episode_reward

        torch.save(policy.network.state_dict(), "MACFN_spread.pkl")
        writer.add_scalar("MACFN_Spread_reward", avg_test_episode_reward / repeat_episode_num, global_step=frame_idx * max_episode_steps)

