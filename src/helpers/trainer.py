# Imports inside the library
from ..config import SAVE_VIDEO
from ..helpers import convert_numpy_to_video

def DQNTrainer(agent, env, episodes, render=False):
    for episode in range(episodes): # TODO: Include options, as many episodes as necessary to fullfill a policy
        state, _ = env.reset()
        terminated = False
        truncated = False
        total_reward = 0
        n_steps = 0

        while not terminated and not truncated:
            if render:
                env.render()

            action = agent.act(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            agent.remember(state, action, reward, next_state, terminated, truncated)
            agent.learn()

            total_reward += reward
            state = next_state

            n_steps += n_steps

        print(f"Episode {episode + 1}/{episodes}, Number of steps in the episode: {n_steps}, Total Reward: {total_reward}")


def DQNInference(agent, env, episodes, steps,render=False):
    for episode in range(episodes):
        state = env.reset()
        terminated = False
        truncated = False
        total_reward = 0
        frames_list = []

        #while not terminated and not truncated:
        for _ in range(steps):
            if render:
                frames_list.append(env.render())
            # Choose action
            action = agent.act(state)
            # Take action in the environment
            next_state, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward

            state = next_state

        convert_numpy_to_video(frames_list, SAVE_VIDEO)

        print(f'Episode {episode + 1}: Total Reward = {total_reward}')
