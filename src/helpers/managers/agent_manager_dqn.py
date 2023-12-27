from tqdm import tqdm

from config import SAVE_VIDEO
from src.helpers import convert_numpy_to_video
from src.helpers.managers.agent_manager_interface import AgentManagerInterface
from src.algorithms.agent_interface import Agent


class DQNAgentManager(AgentManagerInterface):

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
                    n_steps: int = None
                    ):

        while not self.done or self.step == n_steps:

            if self.render:
                self.frames_list.append(self.env.render())

            # Act in the environment
            action = self.agent.act(self.state)
            next_state, reward, self.terminated, self.truncated, _ = self.env.step(action)

            if train_mode:
                self.agent.remember(self.state, action, reward, next_state, self.terminated, self.truncated)
                self.agent.learn(self.episode)

            # End episode
            if self.terminated or self.truncated:
                self.done = True

            # Update reward and state
            self.total_reward += reward
            self.state = next_state

            # Update counter of steps in episode
            self.step += 1

    def train(self):

        for e in tqdm(range(self.n_episodes)):
            self.episode = e
            self.init_episode()
            self.run_episode(train_mode=True)

            tqdm.set_description(
                f"Number of steps in the episode: {str(self.step)}, Total Reward: {str(self.total_reward)}"
            )

        return self.agent, self.all_total_rewards

    def inference(self, n_steps):

        self.run_episode(n_steps=n_steps)
        if self.render:
            convert_numpy_to_video(self.frames_list, SAVE_VIDEO)
