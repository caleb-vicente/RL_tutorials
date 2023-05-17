import os
import datetime

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

from .RLinterfece import RLAlgorithm
from ..config import SAVE_MODEL


class PolicyNetwork(nn.Module):
    def __init__(self, n_inputs, n_outputs):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(n_inputs, 64)
        self.fc2 = nn.Linear(64, n_outputs)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return F.softmax(self.fc2(x), dim=-1)


class REINFORCEAgent(RLAlgorithm):
    def __init__(self, n_inputs, n_outputs, learning_rate=0.01):
        self.policy = PolicyNetwork(n_inputs, n_outputs)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)
        self.memory = []

    def act(self, state):
        state = torch.tensor(state, dtype=torch.float)
        probs = self.policy(state)
        action = torch.multinomial(probs, 1).item()
        return action

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward))

    def learn(self):
        R = 0
        rewards = []

        # calculate discounted rewards
        for _, _, reward in reversed(self.memory):
            R = reward + 0.99 * R
            rewards.insert(0, R)

        # normalize the rewards
        rewards = torch.tensor(rewards)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-9)

        # backpropagate
        for (state, action, _), r in zip(self.memory, rewards):
            self.optimizer.zero_grad()
            state = torch.tensor(state, dtype=torch.float)
            probs = self.policy(state)
            log_prob = torch.log(probs[action])
            loss = -log_prob * r
            loss.backward()
            self.optimizer.step()

        self.memory = []

    def save(self, path=SAVE_MODEL):
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        filename = f"REINFORCE_agent_{timestamp}.pt"
        filepath = path + filename

        # Create directory if it doesn't exist
        os.makedirs(path, exist_ok=True)

        torch.save(self.policy.state_dict(), filepath)
        print(f"Checkpoint saved in {filepath}")

        return filepath

    def load(self, path):
        self.policy.load_state_dict(torch.load(path))
