import datetime
import os
import copy

import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical

from src.algorithms.agent_interface import Agent
from config import SAVE_MODEL


# class PPOModel(nn.Module):
#     def __init__(self, state_dim, action_dim, has_continuous_action_space, action_std_init=0):
#         super(PPOModel, self).__init__()
#
#         self.has_continuous_action_space = has_continuous_action_space
#
#         # Continuous action space
#         if has_continuous_action_space:
#             self.action_dim = action_dim
#             self.action_var = torch.full((action_dim,), action_std_init * action_std_init)
#             self.actor = nn.Sequential(
#                 nn.Linear(state_dim, 64),
#                 nn.Tanh(),
#                 nn.Linear(64, 64),
#                 nn.Tanh(),
#                 nn.Linear(64, action_dim),
#                 nn.Tanh()  # Outputs raw action values, not probabilities
#             )
#         # Discrete action space
#         else:
#             self.actor = nn.Sequential(
#                 nn.Linear(state_dim, 64),
#                 nn.Tanh(),
#                 nn.Linear(64, 64),
#                 nn.Tanh(),
#                 nn.Linear(64, action_dim),
#                 nn.Softmax(dim=-1)  # Outputs action probabilities
#             )
#
#         # Critic
#         self.critic = nn.Sequential(
#             nn.Linear(state_dim, 64),
#             nn.Tanh(),
#             nn.Linear(64, 64),
#             nn.Tanh(),
#             nn.Linear(64, 1)  # Outputs a value estimate
#         )
#
#     def forward(self, state):
#         # Actor forward pass
#         if self.has_continuous_action_space:
#             # Continuous action space: return raw action values
#             action_mean = self.actor(state)
#             action_var = self.action_var.expand_as(action_mean)
#             action_probs = torch.distributions.Normal(action_mean, action_var.sqrt())
#         else:
#             # Discrete action space: return action probabilities
#             action_probs = self.actor(state)
#
#         # Critic forward pass
#         state_value = self.critic(state)
#         return action_probs, state_value


class PPOAgent(Agent):
    def __init__(self, model, gamma=0.99, lr_actor=1e-3, lr_critic=1e-3, eps_clip=0.2, ppo_epochs=10, critic_discount=0.5, entropy_beta=0.01):
        super().__init__()
        self.has_continuous_action_space = False
        self.model = model
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.ppo_epochs = ppo_epochs
        self.critic_discount = critic_discount
        self.entropy_beta = entropy_beta
        self.optimizer = torch.optim.Adam([
                        {'params': self.model.actor.parameters(), 'lr': lr_actor},
                        {'params': self.model.critic.parameters(), 'lr': lr_critic}
                    ])
        self.model_old = copy.deepcopy(model)
        self.memory = []

    def act(self, state):

        state = torch.Tensor(state)
        action_probs, state_val = self.model_old(state)

        if self.has_continuous_action_space:
            cov_mat = torch.diag(self.action_var).unsqueeze(dim=0)
            dist = MultivariateNormal(action_probs, cov_mat)
        else:
            dist = Categorical(action_probs)

        action = dist.sample()
        action_logprob = dist.log_prob(action)

        return action.detach(), action_logprob.detach(), state_val.detach()

    def evaluate(self, state, action):

        state = torch.Tensor(state)
        action_probs, state_values = self.model(state)

        if self.has_continuous_action_space:
            action_var = self.action_var.expand_as(action_probs)
            cov_mat = torch.diag_embed(action_var)
            dist = MultivariateNormal(action_probs, cov_mat)

            # for single action continuous environments
            if self.action_dim == 1:
                action = action.reshape(-1, self.action_dim)

        else:
            dist = Categorical(action_probs)

        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()

        return action_logprobs, state_values, dist_entropy

    def remember(self, state, action, reward, next_state, done, log_prob=None, value=None):
        self.memory.append((state, action, reward, next_state, done, log_prob, value))

    def _loss(self, policy_ratio, advantages, state_values, returns, dist_entropy):
        # Calculate clipped and unclipped objectives
        clipped_objective = - torch.min(policy_ratio * advantages,
                                        torch.clamp(policy_ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantages)

        # Calculate the total loss
        critic_loss = F.mse_loss(state_values.squeeze(), returns)
        loss = clipped_objective + self.critic_discount * critic_loss - self.entropy_beta * dist_entropy

        return loss

    def _update_agent(self, loss):
        self.optimizer.zero_grad()
        loss.mean().backward()
        self.optimizer.step()

    def learn(self, n_steps):

        states, actions, rewards, next_states, dones, old_log_probs, values = zip(*self.memory)

        # Convert lists to tensors
        states = torch.stack([torch.from_numpy(array) for array in states]).detach()
        actions = torch.tensor(actions).detach()
        rewards = list(rewards)
        dones = list(dones)
        old_log_probs = torch.stack(old_log_probs).detach()
        values = torch.stack(values).detach()

        # Calculate the returns and advantages
        returns, advantages = self._calculate_returns_advantages(rewards, values, dones)

        for _ in range(self.ppo_epochs):
            # Obtain log probabilities and state values for current policy
            new_log_probs, new_state_values, dist_entropy = self.evaluate(states, actions)

            # Calculate the policy ratio
            policy_ratio = torch.exp(new_log_probs - old_log_probs.detach())

            # Calculate loss and update agent
            loss = self._loss(policy_ratio, advantages, new_state_values, returns, dist_entropy)
            self._update_agent(loss)

        self.model_old.load_state_dict(self.model.state_dict())

        self.memory.clear()

    def _calculate_returns_advantages(self, rewards, values, dones, tau=0.95):

        returns = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(rewards), reversed(dones)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            returns.insert(0, discounted_reward)

        # Normalizing the rewards
        returns = torch.tensor(returns, dtype=torch.float32)
        returns = (returns - returns.mean()) / (returns.std() + 1e-7)

        advantages = returns.detach() - values.squeeze().detach()
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