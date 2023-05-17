import matplotlib.pyplot as plt
import gymnasium
from gymnasium.wrappers import TimeLimit

from src import DNN, REINFORCEAgent, REINFORCETrainer

env = gymnasium.make("CartPole-v1")
env = TimeLimit(env, max_episode_steps=1000)

input_dim = env.observation_space.shape[0]
output_dim = env.action_space.n

agent = REINFORCEAgent(input_dim, output_dim)
agent, all_total_rewards = REINFORCETrainer(agent, env, episodes=1000)
checkpoint_path = agent.save()

plt.plot(all_total_rewards)
plt.show()

# Example of inference
# env = gymnasium.make("CartPole-v1", render_mode='rgb_array')
# model = DNN(input_dim=input_dim, output_dim=output_dim, layers_sizes=[50, 100], activation_function='relu')
# agent = DQNAgent(env, model, epsilon_start=0)
# agent.load(checkpoint_path)
# DQNInference(agent, env, episodes=1, steps=500, render=True)
