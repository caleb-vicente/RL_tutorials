# Imports inside the library
from ..config import SAVE_VIDEO
from ..helpers import convert_numpy_to_video


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

            if terminated == True | truncated == True:
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


def REINFORCETrainer(agent, env, episodes, render=False):
    # Initialize the environment and the agent

    all_total_rewards = []

    for i_episode in range(episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False
        counter = 0

        while True:
            action = agent.act(state)
            next_state, reward, terminated, truncated, _ = env.step(action)

            if terminated == True | truncated == True:
                done = True

            agent.remember(state, action, counter + 1, next_state, done)
            state = next_state
            total_reward += reward

            if done:
                break

            counter += 1
            total_reward = counter

        agent.learn()
        all_total_rewards.append(total_reward)
        print(f"Episode: {i_episode + 1}, Total reward: {total_reward}")

    return agent, all_total_rewards


def REINFORCEInference(agent, env, episodes, steps, render=False):
    # Initialize the environment and the agent


    for i_episode in range(episodes):
        total_reward = 0
        done = False
        frames_list = []

        state, _ = env.reset()
        for t in range(steps):
            if render:
                frames_list.append(env.render())

            action = agent.act(state)
            state, reward, truncated, terminated, info = env.step(action)

            if terminated == True | truncated == True:
                done = True

            total_reward += reward

            if done:
                print("Episode finished after {} timesteps".format(t + 1))
                break

        convert_numpy_to_video(frames_list, SAVE_VIDEO)

        print(f'Episode {i_episode}: Total Reward = {total_reward}')