import matplotlib.pyplot as plt
import gymnasium
from gymnasium.wrappers import TimeLimit

from src import DNN, REINFORCEAgent, REINFORCETrainer, REINFORCEInference
from src import moving_average

env = gymnasium.make("CartPole-v1")
env = TimeLimit(env, max_episode_steps=1000)

input_dim = env.observation_space.shape[0]
output_dim = env.action_space.n

agent = REINFORCEAgent(input_dim, output_dim)
agent, all_total_rewards = REINFORCETrainer(agent, env, episodes=1000)
checkpoint_path = agent.save()


average_all_total_rewards = moving_average(all_total_rewards, 50)
plt.plot(average_all_total_rewards)
plt.show()

# Example of inference
env = gymnasium.make("CartPole-v1", render_mode='rgb_array')
agent = REINFORCEAgent(input_dim, output_dim)
agent.load(checkpoint_path)
REINFORCEInference(agent, env, episodes=1, steps=500, render=True)
