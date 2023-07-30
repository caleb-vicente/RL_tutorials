import matplotlib.pyplot as plt
import gymnasium
from gymnasium.wrappers import TimeLimit

from src import Structure2Vec, DQNAgent, DQNTrainer, DQNInference
from src import DeliveryEnv_structure2vec
from src import moving_average

#############################################################
#                   EXAMPLE OF TRAINING                     #
#############################################################
n_stops = 20
env = DeliveryEnv_structure2vec(n_stops=20,
                                weight_target=1,
                                weight_opt=1,
                                flag_action_mask=True)

input_dim = env.observation_space.shape[0]
# input_dim = env.observation_space.n # If the observation space is discrete
output_dim = env.action_space.n

nodes_features = None
nodes_cost = None

model = Structure2Vec(env.dist_matrix, num_features=1, embed_dim=64, num_updates=4, num_nodes=n_stops)
agent = DQNAgent(env, model, lr=0.001, gamma=0.1, epsilon_start=1, epsilon_end=0.05, epsilon_decay=0.95,
                 buffer_size='inf', batch_size=1, update_target_freq=10, flag_target=True)
# TODO: At this moment structure2vec only works with batch size=1
# TODO: For the learning rate, we use exponential decay after a certain number of steps, where the decay factor is fixed to 0.95

agent, all_total_rewards = DQNTrainer(agent, env, episodes=5000)
checkpoint_path = agent.save()

average_all_total_rewards = moving_average(all_total_rewards, 50)
plt.plot(average_all_total_rewards)
plt.show()

#############################################################
#                   EXAMPLE OF INFERENCE                    #
#############################################################


# env = DeliveryEnv(n_stops=20, flag_action_mask=True)

input_dim = env.observation_space.shape[0]
output_dim = env.action_space.n

# checkpoint_path = r"C:\Users\caleb\Documents\  projects\reinforcement_learning\RL_tutorials\checkpoints\REINFORCE_agent_20230521121143.pt"

# model = PolicyNetwork(input_dim, output_dim)
# agent = REINFORCEAgent(model,
#                        learning_rate=0.0009,
#                        gamma=0.99)
# agent.load(checkpoint_path)
# REINFORCEInference(agent, env, episodes=1, steps=500, render=True, type_render='image',flag_mask=True)
#
