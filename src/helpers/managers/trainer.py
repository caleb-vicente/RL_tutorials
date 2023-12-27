import torch

# Imports inside the library
from config import SAVE_VIDEO
from src.helpers import convert_numpy_to_video


def DQNTrainer(agent, env, episodes, render=False):
    all_total_rewards = []

    for episode in range(episodes):  # TODO: Include options, as many episodes as necessary to fullfill a policy
        state, _ = env.reset()
        terminated = False
        truncated = False
        done = False
        total_reward = 0
        n_steps = 1

        while not done:
            if render:
                env.render()

            action = agent.act(state)  # TODO: It seems that is not doing batch processing when acting, only learning
            next_state, reward, terminated, truncated, _ = env.step(action)
            agent.remember(state, action, reward, next_state, terminated, truncated)
            agent.learn(episode)

            if terminated == True or truncated == True:
                done = True

            total_reward += reward
            state = next_state

            n_steps += 1

        all_total_rewards.append((total_reward))
        print(
            f"Episode {episode + 1}/{episodes}, Number of steps in the episode: {n_steps}, Total Reward: {total_reward}")
        test = 'test'

    return agent, all_total_rewards


def DQNInference(agent, env, episodes, steps, render=False):
    for episode in range(episodes):
        state, _ = env.reset()
        terminated = False
        truncated = False
        total_reward = 0
        frames_list = []

        # while not terminated and not truncated:
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


def REINFORCETrainer(agent, env, episodes, max_steps_episode=200, flag_mask=False):

    # Initialize the environment and the agent
    all_total_rewards = []

    for i_episode in range(episodes):
        state, info = env.reset()
        total_reward = 0
        done = False
        counter = 0

        while not done:

            if flag_mask:
                agent.set_mask(torch.tensor(info['mask'])) # TODO: set mask should be included in the reinforcement algorithm

            action = agent.act(state)
            next_state, reward, terminated, truncated, info = env.step(action)

            if terminated == True or truncated == True:
                done = True

            total_reward += reward
            counter += 1
            agent.remember(state, action, total_reward, next_state, done)
            state = next_state

            if counter > max_steps_episode:
                print('reach max episode')
                break

        agent.learn()
        all_total_rewards.append(total_reward)
        print(f"Episode: {i_episode + 1}, Total reward: {total_reward}")

    return agent, all_total_rewards


def REINFORCEInference(agent, env, episodes, steps, render=False, type_render='image', flag_mask=False):
    # Initialize the environment and the agent

    for i_episode in range(episodes):

        # while not terminated and not truncated:
        state, info = env.reset()
        done = False
        total_reward = 0
        frames_list = []

        while not done:

            if render and type_render=='video':
                frames_list.append(env.render())

            # If mask is available, set it for the agent
            if flag_mask:
                agent.set_mask(torch.tensor(info['mask']))

            action = agent.act(state)
            state, reward, truncated, terminated , info = env.step(action)
            total_reward += reward

            if terminated == True or truncated == True:
                done = True

        if render and type_render=='video':
            convert_numpy_to_video(frames_list, SAVE_VIDEO)
        elif render and type_render=='image':
            env.render()

        print(f'Episode {i_episode}: Total Reward = {total_reward}')
