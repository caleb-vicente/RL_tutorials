from tqdm import tqdm

from config import SAVE_VIDEO
from src.helpers import convert_numpy_to_video
from src.helpers.managers.agent_manager_interface import AgentManagerInterface
from src.algorithms.agent_interface import Agent


class A2CAgentManager(AgentManagerInterface):

    def __init__(self,
                 agent: Agent,
                 env: object,
                 n_episodes: int,
                 render: bool = False
                 ):

        # config class parameters
        self.agent = agent
        self.env = env
        self.n_episodes = n_episodes
        self.render = render

        # init parameter episode
        self.all_total_rewards = None
        self.state = None
        self.terminated = None
        self.truncated = None
        self.done = None
        self.total_reward = None
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
        self.step = 1
        self.episode = 0
        self.frames_list = []

    def run_episode(self,
                    train_mode: bool = False,
                    n_steps: int = None,
                    reward_end_episode: int = 0,
                    ):

        while not self.done or self.step == n_steps:

            if self.render:
                self.frames_list.append(self.env.render())

            # Act in the environment
            action, value, log_prob = self.agent.act(self.state)
            next_state, reward, self.terminated, self.truncated, _ = self.env.step(action.detach().numpy())

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

        if train_mode:
            self.agent.learn()

    def train(self, reward_end_episode=0):

        pbar = tqdm(range(self.n_episodes))
        for e in pbar:
            self.env.reset()
            self.episode = e
            self.init_episode()
            self.run_episode(train_mode=True,
                             reward_end_episode=reward_end_episode)

            pbar.set_description(f"Number of steps in the episode: {self.step}, Total Reward: {self.total_reward}")

        return self.agent, self.all_total_rewards

    def inference(self, n_steps):

        self.init_episode()
        self.run_episode(n_steps=n_steps)
        if self.render:
            convert_numpy_to_video(self.agent.__class__.__name__, self.frames_list, SAVE_VIDEO)
