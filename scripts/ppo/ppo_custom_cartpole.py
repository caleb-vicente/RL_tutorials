import gymnasium
from gymnasium.wrappers import TimeLimit
import torch
from torch import nn
from torch.nn import functional as F

from src import PPOAgent, PPOAgentManager


class PPOModel(nn.Module):
    def __init__(self, state_dim, action_dim, has_continuous_action_space, action_std_init=0):
        super(PPOModel, self).__init__()

        self.has_continuous_action_space = has_continuous_action_space

        # Continuous action space
        if has_continuous_action_space:
            self.action_dim = action_dim
            self.action_var = torch.full((action_dim,), action_std_init * action_std_init)
            self.actor = nn.Sequential(
                nn.Linear(state_dim, 64),
                nn.Tanh(),
                nn.Linear(64, 64),
                nn.Tanh(),
                nn.Linear(64, action_dim),
                nn.Tanh()  # Outputs raw action values, not probabilities
            )
        # Discrete action space
        else:
            self.actor = nn.Sequential(
                nn.Linear(state_dim, 64),
                nn.Tanh(),
                nn.Linear(64, 64),
                nn.Tanh(),
                nn.Linear(64, action_dim),
                nn.Softmax(dim=-1)  # Outputs action probabilities
            )

        # Critic
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)  # Outputs a value estimate
        )

    def forward(self, state):
        # Actor forward pass
        if self.has_continuous_action_space:
            # Continuous action space: return raw action values
            action_mean = self.actor(state)
            action_var = self.action_var.expand_as(action_mean)
            action_probs = torch.distributions.Normal(action_mean, action_var.sqrt())
        else:
            # Discrete action space: return action probabilities
            action_probs = self.actor(state)

        # Critic forward pass
        state_value = self.critic(state)
        return action_probs, state_value


def main():

    env = gymnasium.make("CartPole-v1", render_mode='rgb_array')
    env = TimeLimit(env, max_episode_steps=1000)
    input_dim = env.observation_space.shape[0]
    output_dim = env.action_space.n

    # ------------ Create model ---------------
    model = PPOModel(state_dim=input_dim,
                     action_dim=output_dim,
                     has_continuous_action_space=False)
    # ------------------------------------------

    # ------------ Create agent ----------------
    agent = PPOAgent(
        model,
        gamma=0.99,
        lr_actor=0.0003,
        lr_critic=0.001,
        eps_clip=0.2,
        ppo_epochs=40,
        critic_discount=0.5,
        entropy_beta=0.01
    )

    agent_manager = PPOAgentManager(
        agent=agent,
        env=env,
        n_episodes=500,
        n_processes=4,
        render=False
    )

    agent, all_total_rewards = agent_manager.train(
        n_steps=None,
        reward_end_episode=0
    )

    agent_path = agent.save()

    # ---------- Inference ----------------------
    print(f"Retrieving the agent in {agent_path}")
    agent_saved = PPOAgent(model=model).load(agent_path)

    agent_manager_saved = PPOAgentManager(
        agent=agent_saved,
        env=env,
        n_episodes=10,
        render=True
    )
    agent_manager_saved.inference(n_steps=300)


if __name__ == '__main__':
    main()