import gymnasium
from src import DNN, DQNAgent, DQNTrainer, DQNInference

# Example of training
env = gymnasium.make("CartPole-v1") # TODO: How can I modify the number of maxiumum steps
input_dim = env.observation_space.shape[0]
output_dim = env.action_space.n
model = DNN(input_dim, output_dim)
agent = DQNAgent(env, model)
agent = DQNTrainer(agent, env, episodes=500)
checkpoint_path = agent.save()

# Example of inference
env = gymnasium.make("CartPole-v1", render_mode='rgb_array')
model = DNN(env.observation_space.shape[0], env.action_space.n)
agent = DQNAgent(env, model, epsilon_start=0)
agent.load(checkpoint_path)
DQNInference(agent, env, episodes=1, steps=300, render=True)