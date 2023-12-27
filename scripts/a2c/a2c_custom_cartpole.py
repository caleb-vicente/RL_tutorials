import gymnasium
import matplotlib.pyplot as plt
from src import ActorCriticAgent, A2CAgentManager
import gymnasium
from gymnasium.wrappers import TimeLimit
import torch
from torch import nn
from torch.nn import functional as F

env = gymnasium.make("CartPole-v1", render_mode='rgb_array')
env = TimeLimit(env, max_episode_steps=1000)
input_dim = env.observation_space.shape[0]
output_dim = env.action_space.n


# ------------ Create model ---------------
class ActorCriticModel(nn.Module):
    def __init__(self):
        super(ActorCriticModel, self).__init__()
        self.l1 = nn.Linear(4, 50)
        self.l2 = nn.Linear(50, 100)
        self.actor_lin1 = nn.Linear(100, 2)
        self.l3 = nn.Linear(100, 50)
        self.critic_lin1 = nn.Linear(50, 1)

    def forward(self, x):
        x = F.normalize(x, dim=0)
        y = F.relu(self.l1(x))
        y = F.relu(self.l2(y))
        actor = F.log_softmax(self.actor_lin1(y), dim=0)
        c = F.relu(self.l3(y.detach()))
        critic = torch.tanh(self.critic_lin1(c))
        return actor, critic


model = ActorCriticModel()
# ------------------------------------------

# ------------ Create agent ----------------
agent = ActorCriticAgent(
    model,
    gamma=0.9,
    lr=1e-3,
    clc=0.1
)

agent_manager = A2CAgentManager(
    agent=agent,
    env=env,
    n_episodes=1000,
    render=False
)

agent, all_total_rewards = agent_manager.train(
    reward_end_episode=-10
)

agent_path = agent.save()

# ---------- Inference ----------------------
print(f"Retrieving the agent in {agent_path}")
agent_saved = ActorCriticAgent(model=model).load(agent_path)

agent_manager_saved = A2CAgentManager(
    agent=agent_saved,
    env=env,
    n_episodes=10,
    render=True
)
agent_manager_saved.inference(n_steps=300)
