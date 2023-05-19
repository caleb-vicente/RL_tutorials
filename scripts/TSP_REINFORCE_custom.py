import matplotlib.pyplot as plt
import gymnasium
from gymnasium.wrappers import TimeLimit

from src import PolicyNetwork, REINFORCEAgent, REINFORCETrainer, REINFORCEInference
from src import DeliveryEnv
from src import moving_average

#############################################################
#                   EXAMPLE OF TRAINING                     #
#############################################################

# env = DeliveryEnv(n_stops=5)
#
# input_dim = env.observation_space.shape[0]
# output_dim = env.action_space.n
#
# model = PolicyNetwork(input_dim, output_dim)
# agent = REINFORCEAgent(model,
#                        learning_rate=0.002,
#                        gamma=0.99)
#
# agent, all_total_rewards = REINFORCETrainer(agent, env, episodes=1000)
# checkpoint_path = agent.save()
#
# average_all_total_rewards = moving_average(all_total_rewards, 50)
# plt.plot(average_all_total_rewards)
# plt.show()

#############################################################
#                   EXAMPLE OF INFERENCE                    #
#############################################################


env = DeliveryEnv(n_stops=5)

input_dim = env.observation_space.shape[0]
output_dim = env.action_space.n

checkpoint_path = r"C:\Users\caleb\Documents\projects\reinforcement_learning\RL_tutorials\checkpoints\REINFORCE_agent_20230519185307.pt"

model = PolicyNetwork(input_dim, output_dim)
agent = REINFORCEAgent(model,
                       learning_rate=0.002,
                       gamma=0.99)
agent.load(checkpoint_path)
REINFORCEInference(agent, env, episodes=1, steps=500, render=False)
