import datetime
import os

import torch
import torch.nn.functional as F

from src.algorithms.agent_interface import Agent
from config import SAVE_MODEL


class PPOAgent(Agent):
    def __init__(self, model, gamma=0.99, lr=1e-3, clip_param=0.2, ppo_epochs=10, critic_discount=0.5, entropy_beta=0.01):
        super().__init__()
        self.model = model
        self.gamma = gamma
        self.clip_param = clip_param
        self.ppo_epochs = ppo_epochs
        self.critic_discount = critic_discount
        self.entropy_beta = entropy_beta
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.memory = []

    def act(self, state, action=None):
        state = torch.tensor(state, dtype=torch.float32).detach()
        action_probs, value = self.model(state)
        dist = torch.distributions.Categorical(logits=action_probs)
        if action is None:
            action = dist.sample()
        log_prob = dist.log_prob(action)
        dist_entropy = dist.entropy()
        return action, value, log_prob, dist_entropy

    def _evaluate_policy(self, states, actions):
        action_probs, state_values = self.model(states)
        dist = torch.distributions.Categorical(logits=action_probs)
        log_probs = dist.log_prob(actions)
        return log_probs, state_values

    def remember(self, state, action, reward, next_state, done, log_prob=None, value=None):
        self.memory.append((state, action, reward, next_state, done, log_prob, value))

    def _loss(self, policy_ratio, advantages, state_values, returns, dist_entropy):
        # Calculate clipped and unclipped objectives
        clipped_objective = torch.min(policy_ratio * advantages,
                                      torch.clamp(policy_ratio, 1 - self.clip_param, 1 + self.clip_param) * advantages)

        # Calculate the total loss
        actor_loss = - clipped_objective
        critic_loss = F.mse_loss(state_values.squeeze(), returns)
        loss = actor_loss + self.critic_discount * critic_loss + self.entropy_beta * dist_entropy

        return loss

    def _update_agent(self, loss):
        self.optimizer.zero_grad()
        loss.mean().backward()
        self.optimizer.step()

    def learn(self, n_steps):
        if n_steps is not None:
            if len(self.memory) < n_steps:
                return

        states, actions, rewards, next_states, dones, old_log_probs, values = zip(*self.memory)

        # Convert lists to tensors
        states = torch.stack([torch.from_numpy(array) for array in states]).detach()
        actions = torch.tensor(actions).detach()
        rewards = torch.tensor(rewards).detach()
        dones = torch.tensor(dones).detach()
        old_log_probs = torch.stack(old_log_probs).detach()
        values = torch.stack(values).detach()

        # Calculate the returns and advantages
        returns, advantages = self._calculate_returns_advantages(rewards, values, dones)

        for _ in range(self.ppo_epochs):
            # Obtain log probabilities and state values for current policy
            _, new_state_values, new_log_probs, dist_entropy = self.act(states, actions)

            # Calculate the policy ratio
            policy_ratio = torch.exp(new_log_probs - old_log_probs.detach())

            # Calculate loss and update agent
            loss = self._loss(policy_ratio, advantages, new_state_values, returns, dist_entropy)
            self._update_agent(loss)

        self.memory.clear()

    def _calculate_returns_advantages(self, rewards, values, dones, tau=0.95):
        returns = []
        gae = 0
        values = values.squeeze()
        for step in reversed(range(len(rewards))):
            if dones[step]:
                discounted_reward = 0
            else:
                discounted_reward = self.gamma * values[step + 1] * (1 - dones[step].item()) - values[step]
            delta = rewards[step] + discounted_reward
            gae = delta + self.gamma * tau * (1 - dones[step].item()) * gae
            returns.insert(0, gae)

        returns = torch.tensor(returns)
        advantages = returns - values
        return returns, advantages

    def save(self, path=SAVE_MODEL):
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        filename = f"ppo_agent_{timestamp}.pt"
        filepath = path + filename

        # Create directory if it doesn't exist
        os.makedirs(path, exist_ok=True)

        torch.save(self.model.state_dict(), filepath)
        print(f"Checkpoint saved in {filepath}")

        return filepath

    def load(self, path):
        self.model.load_state_dict(torch.load(path))
        return self