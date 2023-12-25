import torch
import torch.nn.functional as F

from src.algorithms.agent_interface import Agent


class ActorCriticAgent(Agent):
    def __init__(self, actor_model, critic_model, gamma=0.99, lr=1e-3, clc=0.1):
        self.actor_model = actor_model
        self.critic_model = critic_model
        self.gamma = gamma
        self.optimizer = torch.optim.Adam(
            list(self.actor_model.parameters()) + list(self.critic_model.parameters()), lr=lr
        )
        self.memory = []
        self.clc = clc

    def act(self, state):
        state = torch.tensor(state, dtype=torch.float32)
        action_probs = self.actor_model(state)
        action = torch.multinomial(action_probs, 1).item()
        return action

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def loss(self, states, values, returns, actions):
        # Critic loss
        critic_loss = F.mse_loss(values, returns.detach())

        # Actor loss
        action_probs = self.actor_model(states)
        advantages = returns - values.detach()
        actor_loss = -(torch.log(action_probs.gather(1, actions.unsqueeze(1))) * advantages).mean()

        # Total loss
        self.optimizer.zero_grad()
        loss = actor_loss + self.clc * critic_loss
        loss.backward()
        self.optimizer.step()
        self.memory.clear()

    def learn(self):
        if not self.memory:
            return

        # Unpack the experiences. Memory only contains one episode
        states, actions, rewards, next_states, dones = zip(*self.memory)

        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.long)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        next_states = torch.tensor(next_states, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32)

        # Compute value targets
        values = self.critic_model(states)
        next_values = self.critic_model(next_states)
        returns = rewards + self.gamma * next_values * (1 - dones)

        # Update parameters
        self.update_critic(values, returns)
        self.update_actor(states, values, returns, actions)

    def save(self, filename):
        torch.save({
            'actor_model_state_dict': self.actor_model.state_dict(),
            'critic_model_state_dict': self.critic_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }, filename)

    def load(self, filename):
        checkpoint = torch.load(filename)
        self.actor_model.load_state_dict(checkpoint['actor_model_state_dict'])
        self.critic_model.load_state_dict(checkpoint['critic_model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
