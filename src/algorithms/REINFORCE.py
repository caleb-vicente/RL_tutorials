import os
import datetime

import torch
import torch.optim as optim
import numpy as np

from .RLinterfece import RLAlgorithm
from ..config import SAVE_MODEL


class REINFORCEAgent(RLAlgorithm):
    def __init__(self, model, learning_rate=0.002, gamma=0.99):
        self.policy = model
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)
        self.gamma = gamma
        self.memory = []
        self.mask = None

    def set_mask(self, mask):
        self.mask = mask

    def act(self, state):
        act_prob = self.policy(torch.from_numpy(state).float())

        if self.mask is not None:
            act_prob = act_prob * self.mask
            act_prob /= act_prob.sum()

        p = act_prob.data.numpy().squeeze()

        action = np.random.choice(np.arange(act_prob.shape[0]), p=p)
        self.mask = None
        return action

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward))

    def discount_rewards(self, rewards):
        lenr = len(rewards)
        disc_return = torch.pow(self.gamma, torch.arange(lenr).float()) * rewards
        disc_return /= disc_return.max()
        return disc_return

    def loss_fn(self, preds, r):
        return -1 * torch.sum(r * torch.log(preds))

    def learn(self):
        reward_batch = torch.Tensor([r for (s, a, r) in self.memory]).flip(dims=(0,))
        disc_rewards = self.discount_rewards(reward_batch)
        state_batch = torch.Tensor([s for (s, a, r) in self.memory])
        action_batch = torch.Tensor([a for (s, a, r) in self.memory])

        pred_batch = self.policy(state_batch)
        if self.mask is not None:
            pred_batch = pred_batch * self.mask

        prob_batch = pred_batch.gather(dim=1, index=action_batch.long().view(-1, 1)).squeeze()
        loss = self.loss_fn(prob_batch, disc_rewards)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.memory = []
        self.mask = None

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
