# Adapted from https://docs.pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
import sys
import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math
from collections import namedtuple, deque
from itertools import count
import random
from game import GameState

from typing import List, Any

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory:

    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQNPongAgent:

    def __init__(self, layers: int = 1, dimensions: int = 512):
        self.device: str = "cpu"
        accelerator = torch.accelerator.current_accelerator()
        if torch.accelerator.is_available() and accelerator is not None:
            self.device = str(accelerator.type)

        self.policy = DQN(layers, dimensions).to(self.device)
        self.target = DQN(layers, dimensions).to(self.device)
        self.target.load_state_dict(self.policy.state_dict())
        self.memory = ReplayMemory(10000)
        self.BATCH_SIZE = 128
        self.GAMMA = 0.99
        self.EPS_START = 0.9
        self.EPS_END = 0.05
        self.EPS_DECAY = 1000
        self.TAU = 0.005
        self.LR = 1e-4
        self.steps_done = 0
        self.episode_durations: List[int] = []
        self.optimizer = optim.AdamW(self.policy.parameters(),
                                     lr=self.LR,
                                     amsgrad=True)

    def train(self):
        env = GameState()
        if torch.cuda.is_available() or torch.backends.mps.is_available():
            num_episodes = 600
        else:
            num_episodes = 50

        for i_episode in range(num_episodes):
            # Initialize the environment and get its state
            env.reset()
            state = env.get_context()
            state = torch.tensor(state,
                                 dtype=torch.float32,
                                 device=self.device).unsqueeze(0)
            for t in count():
                action = self._select_action(state)
                env.update(int(action))
                reward = env.get_score()
                terminated = env.over
                observation = env.get_context()
                #observation, reward, terminated, truncated, _ = env.step(action.item())
                reward = torch.tensor([reward], device=self.device)
                done = terminated

                if terminated:
                    next_state = None
                else:
                    next_state = torch.tensor(observation,
                                              dtype=torch.float32,
                                              device=self.device).unsqueeze(0)

                # Store the transition in memory
                self.memory.push(state, action, next_state, reward)

                # Move to the next state
                state = next_state

                # Perform one step of the optimization (on the policy network)
                self._optimize()

                # Soft update of the target network's weights
                target_net_state_dict = self.target.state_dict()
                policy_net_state_dict = self.policy.state_dict()
                for key in policy_net_state_dict:
                    target_net_state_dict[key] = policy_net_state_dict[
                        key] * self.TAU + target_net_state_dict[key] * (
                            1 - self.TAU)
                self.target.load_state_dict(target_net_state_dict)

                if done:
                    self.episode_durations.append(t + 1)
                    print(f"Episode {i_episode} complete: score {reward}")
                    break

    def _select_action(self, state):
        sample = random.random()
        eps_threshold = self.EPS_END + (
            self.EPS_START - self.EPS_END) * math.exp(
                -1. * self.steps_done / self.EPS_DECAY)
        self.steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                return self.policy(state).max(1).indices.view(1, 1)
        else:
            return torch.tensor([random.sample([0, 1, 2], 1)],
                                device=self.device,
                                dtype=torch.long)

    def _optimize(self):
        if len(self.memory) < self.BATCH_SIZE:
            return

        transitions = self.memory.sample(self.BATCH_SIZE)

        batch = Transition(*zip(*transitions))

        non_final_mask = torch.tensor(tuple(
            map(lambda s: s is not None, batch.next_state)),
                                      device=self.device,
                                      dtype=torch.bool)

        non_final_next_states = torch.cat(
            [s for s in batch.next_state if s is not None])

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        state_action_values = self.policy(state_batch).gather(1, action_batch)

        next_state_values = torch.zeros(self.BATCH_SIZE, device=self.device)

        with torch.no_grad():
            next_state_values[non_final_mask] = self.target(
                non_final_next_states).max(1).values

        # Compute the expected Q values
        expected_state_action_values = (next_state_values *
                                        self.GAMMA) + reward_batch

        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values,
                         expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.policy.parameters(), 100)
        self.optimizer.step()

    def eval(self, state) -> int:
        with torch.no_grad():
            return self.policy(state).max(1).indices.view(1, 1)


class DQN(nn.Module):

    def __init__(self, layers: int = 1, dimension: int = 512):
        super(DQN, self).__init__()
        self.flatten = nn.Flatten()
        internal_layers: List[Any] = []
        for i in range(layers):
            internal_layers.append(nn.Linear(dimension, dimension))
            internal_layers.append(nn.ReLU())
        self.linear_relu_stack = nn.Sequential(nn.Linear(5, dimension),
                                               nn.ReLU(), *internal_layers,
                                               nn.Linear(dimension, 3))

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
