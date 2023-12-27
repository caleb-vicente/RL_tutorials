from tqdm import tqdm
import torch.multiprocessing as mp
from copy import deepcopy

from config import SAVE_VIDEO
from src.helpers import convert_numpy_to_video
from src.helpers.managers.agent_manager_interface import AgentManagerInterface
from src.algorithms.agent_interface import Agent


class A2CAgentManager(AgentManagerInterface):

    def __init__(self,
                 agent: Agent,
                 env: object,
                 n_episodes: int,
                 n_processes: int = 1,
                 render: bool = False
                 ):

        # config class parameters
        self.agent = agent
        self.env = env
        self.n_episodes = int(n_episodes/n_processes)
        self.render = render
        self.n_processes = n_processes

        # init parameter episode
        self.all_total_rewards = None
        self.state = None
        self.terminated = None
        self.truncated = None
        self.done = None
        self.total_reward = None
        self.episode_step = None
        self.step = None
        self.episode = 0
        self.frames_list = []

    def init_episode(self):
        self.all_total_rewards = []
        self.state, _ = self.env.reset()
        self.terminated = False
        self.truncated = False
        self.done = False
        self.total_reward = 0
        self.episode_step = 1
        self.step = 1
        self.episode = 0
        self.frames_list = []

    def run_episode(self,
                    env,
                    train_mode: bool = False,
                    n_steps: int = None,
                    reward_end_episode: int = 0,
                    ):

        while not self.done:

            if self.render:
                self.frames_list.append(env.render())

            # Act in the environment
            action, value, log_prob = self.agent.act(self.state)
            next_state, reward, self.terminated, self.truncated, _ = env.step(action.detach().numpy())

            # End episode
            if self.terminated or self.truncated:
                self.done = True
                reward = reward_end_episode

            if train_mode:
                self.agent.remember(self.state, action, reward, next_state, self.done, log_prob, value)

            # Update reward and state
            self.total_reward += reward
            self.state = next_state

            # Update counter of steps in episode
            self.step += 1
            self.episode_step += 1

            if n_steps is not None and train_mode:
                if self.step == n_steps:
                    self.step = 0
                    self.agent.memory = []
                    self.agent.learn(n_steps=n_steps)

        if train_mode:
            self.agent.learn(n_steps)

    def train(self, n_steps, reward_end_episode=0):
        # Ensure the model is shared between processes
        self.agent.model.share_memory()

        # Create and start processes
        processes = []
        for rank in range(self.n_processes):
            p = mp.Process(target=self._train_process, args=(rank, n_steps, reward_end_episode))
            p.start()
            processes.append(p)

        # Wait for all processes to finish
        for p in processes:
            p.join()

        return self.agent, self.all_total_rewards

    def _train_process(self, rank, n_steps, reward_end_episode=0):

        env = self._clone_env()
        pbar = tqdm(range(self.n_episodes))
        for e in pbar:
            env.reset()
            self.episode = e
            self.init_episode()
            self.run_episode(env, n_steps=n_steps, train_mode=True, reward_end_episode=reward_end_episode)

            pbar.set_description(f"Number of steps in the episode: {self.episode_step}, Total Reward: {self.total_reward}")

    def inference(self, n_steps):

        self.init_episode()
        self.run_episode(self.env, n_steps=n_steps)
        if self.render:
            convert_numpy_to_video(self.agent.__class__.__name__, self.frames_list, SAVE_VIDEO)

    def _clone_env(self):
        return deepcopy(self.env)
