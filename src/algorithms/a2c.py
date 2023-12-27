import datetime
import os

import torch
import torch.nn.functional as F

from src.algorithms.agent_interface import Agent
from config import SAVE_MODEL


class ActorCriticAgent(Agent):
    def __init__(self, model=None, gamma=0.99, lr=1e-3, clc=0.1):
        self.model = model
        self.gamma = gamma
        self.optimizer = torch.optim.Adam(
            list(self.model.parameters()), lr=lr
        )
        self.memory = []
        self.clc = clc

    def act(self, state):
        state = torch.tensor(state, dtype=torch.float32)
        action_probs, value = self.model(state)
        action = torch.distributions.Categorical(logits=action_probs).sample()
        log_prob = torch.distributions.Categorical(logits=action_probs).log_prob(action)
        return action, value, log_prob

    def remember(self, state, action, reward, next_state, done, log_prob=None, value=None):
        self.memory.append((state, action, log_prob, value, reward, next_state, done))

    def loss(self, values, returns, log_probs):

        # Critic loss
        critic_loss = F.mse_loss(values, returns)

        # Actor loss
        advantages = returns - values.detach()
        actor_loss = -(log_probs * advantages).mean()

        loss = actor_loss + self.clc * critic_loss
        return loss

    def update_agent(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.memory.clear()

    def learn(self):
        if not self.memory:
            return

        # Unpack the experiences. Memory only contains one episode
        states, actions, log_probs, values, rewards, next_states, dones = zip(*self.memory)

        actions = torch.tensor(actions, dtype=torch.long).flip(dims=(0,)).view(-1)
        log_probs = torch.stack(log_probs).flip(dims=(0,)).view(-1)
        values = torch.tensor(values, dtype=torch.float32).flip(dims=(0,)).view(-1)
        rewards = torch.tensor(rewards, dtype=torch.float32).flip(dims=(0,)).view(-1)

        returns = []
        ret_ = torch.Tensor([0])
        for r in range(rewards.shape[0]):
            ret_ = rewards[r] + self.gamma * ret_
            returns.append(ret_)
        returns = torch.stack(returns).squeeze()

        # Update parameters
        loss = self.loss(values, returns, log_probs)
        self.update_agent(loss)

    def save(self, path=SAVE_MODEL):
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        filename = f"a2c_agent_{timestamp}.pt"
        filepath = path + filename

        # Create directory if it doesn't exist
        os.makedirs(path, exist_ok=True)

        torch.save(self.model.state_dict(), filepath)
        print(f"Checkpoint saved in {filepath}")

        return filepath

    def load(self, path):
        self.model.load_state_dict(torch.load(path))
        return self
