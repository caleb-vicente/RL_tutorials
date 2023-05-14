import gymnasium
import matplotlib.pyplot as plt
from src import DNN, DQNAgent, DQNTrainer, DQNInference


# Now you can create an instance of your custom environment
import gymnasium
from gymnasium.wrappers import TimeLimit
env = gymnasium.make("CartPole-v1")
env = TimeLimit(env, max_episode_steps=1000)
#env = gymnasium.make('CartPole-v1')

input_dim = env.observation_space.shape[0]
output_dim = env.action_space.n
model = DNN(input_dim=input_dim, output_dim=output_dim, layers_sizes=[50, 100], activation_function='relu')
# CartPole with instant replay
# agent = DQNAgent(env, model, lr=0.001, gamma=0.9, epsilon_start=0.2, epsilon_end=0.01, epsilon_decay=0.99,
#                  buffer_size='inf', batch_size=20, update_target_freq=10, flag_target=False)
# CartPole with target network
agent = DQNAgent(env, model, lr=0.001, gamma=0.9, epsilon_start=0.2, epsilon_end=0.01, epsilon_decay=0.99,
                 buffer_size='inf', batch_size=20, update_target_freq=10, flag_target=True)
# Cart Pole with Double DQN
# agent = DQNAgent(env, model, lr=0.001, gamma=0.9, epsilon_start=0.5, epsilon_end=0.01, epsilon_decay=0.99,
#                  buffer_size=100000, batch_size=20, update_target_freq=10, flag_double=True)

agent, all_total_rewards = DQNTrainer(agent, env, episodes=1000)
checkpoint_path = agent.save()

plt.plot(all_total_rewards)
plt.show()

# Example of inference
env = gymnasium.make("CartPole-v1", render_mode='rgb_array')
model = DNN(input_dim=input_dim, output_dim=output_dim, layers_sizes=[50, 100], activation_function='relu')
agent = DQNAgent(env, model, epsilon_start=0)
agent.load(checkpoint_path)
DQNInference(agent, env, episodes=1, steps=500, render=True)