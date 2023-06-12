import matplotlib.pyplot as plt
import gymnasium
from gymnasium.wrappers import TimeLimit

from src import PolicyNetwork, REINFORCEAgent, REINFORCETrainer, REINFORCEInference
from src import DeliveryEnv, DeliveryEnv_v2, DeliveryEnv_v3
from src import moving_average

#############################################################
#                   EXAMPLE OF TRAINING                     #
#############################################################

env = DeliveryEnv_v3(n_stops=20,
                     weight_target=1,
                     weight_opt=1,
                     flag_action_mask=True)
# env = DeliveryEnv_v2(render_mode='rgb_array',
#                      n_stops=5,
#                      max_box=10,
#                      weight_opt=1,
#                      weight_target=1,
#                      flag_action_mask=False)

input_dim = env.observation_space.shape[0]
# input_dim = env.observation_space.n # If the observation space is discrete
output_dim = env.action_space.n

model = PolicyNetwork(input_dim, output_dim)
agent = REINFORCEAgent(model,
                       learning_rate=0.0009,
                       gamma=0.99)

agent, all_total_rewards = REINFORCETrainer(agent, env, episodes=5000, max_steps_episode=200, flag_mask=True)
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

model = PolicyNetwork(input_dim, output_dim)
agent = REINFORCEAgent(model,
                       learning_rate=0.0009,
                       gamma=0.99)
agent.load(checkpoint_path)
REINFORCEInference(agent, env, episodes=1, steps=500, render=True, type_render='image',flag_mask=True)

