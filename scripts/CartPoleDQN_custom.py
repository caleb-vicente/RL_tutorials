import gymnasium
import matplotlib.pyplot as plt
from src import DNN, DQNAgent, DQNTrainer, DQNInference

# Example of training
env = gymnasium.make("CartPole-v1")
input_dim = env.observation_space.shape[0]
output_dim = env.action_space.n
model = DNN(input_dim, output_dim)
# For CarPole this configuration seems to be working better
agent = DQNAgent(env, model, lr=0.001, gamma=0.9, epsilon_start=0.2, epsilon_end=0.01, epsilon_decay=0.99,
                 buffer_size='inf', batch_size=20, update_target_freq=10, flag_target=False)
# agent = DQNAgent(env, model, lr=1e-3, gamma=0.99, epsilon_start=2.0, epsilon_end=0.01, epsilon_decay=0.99,
#                  buffer_size=100000, batch_size=64, update_target_freq=100, flag_target=True)

agent, all_total_rewards = DQNTrainer(agent, env, episodes=150)
checkpoint_path = agent.save()

plt.plot(all_total_rewards)
plt.show()

# Example of inference
env = gymnasium.make("CartPole-v1", render_mode='rgb_array')
model = DNN(env.observation_space.shape[0], env.action_space.n)
agent = DQNAgent(env, model, epsilon_start=0)
agent.load(checkpoint_path)
DQNInference(agent, env, episodes=1, steps=500, render=True)