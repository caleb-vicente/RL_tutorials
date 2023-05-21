import numpy as np
import gymnasium as gym
from gymnasium import spaces
from scipy.spatial import distance_matrix
import pygame


class DeliveryEnv_v2(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(self,
                 render_mode=None,
                 n_stops=5,
                 max_box=10,
                 weight_opt=1,
                 weight_target=1,
                 flag_action_mask=False):
        super(DeliveryEnv_v2, self).__init__()

        print(f"Initialized Delivery Environment with {n_stops} random stops")

        # Initialization
        self.n_stops = n_stops
        self.action_space = spaces.Discrete(self.n_stops)
        self.observation_space = spaces.Discrete(self.n_stops) # TODO: observation space have to match with
        self.max_box = max_box
        self.stops = []
        self.weight_opt = weight_opt
        self.weight_target = weight_target
        self.grid_size = 50  # Size of the grid used to display the bridge
        self.window_size = self.grid_size * max_box  # The size of the PyGame window

        # Generate stops
        self._generate_stops()

        # Generate noise due to stochastic traffic
        self._get_noise()

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        # If human-rendering is used, `self.window` will be a reference to the window that we draw to.
        self.window = None
        # `self.clock`: a clock that is used to ensure that the environment is rendered at the correct frame rate in
        # human-mode. They will remain `None` until human-mode is used for the first time.
        self.clock = None

        # Variables of control
        self.state = None
        self.truncated = False
        self.terminated = False
        self.flag_action_mask = flag_action_mask
        self.info = None

    def _get_noise(self):
        self.traffic = np.abs(np.random.randn(self.n_stops, self.n_stops)) / 1000

    def get_obs(self):
        obs = {"distance_matrix": self.dist_matrix[self.stops[-1], :].copy(),
                "stochasticity": self.traffic[self.stops[-1], :].copy(), # TODO: Make stochasticity optional
                "current_state": self.memory[self.stops[-1], :].copy()}

        # Normalize
        obs["distance_matrix"] = (obs["distance_matrix"] - obs["distance_matrix"].min()) / \
                                (obs["distance_matrix"].max() - obs["distance_matrix"].min())
        obs["stochasticity"] = (obs["stochasticity"] - obs["stochasticity"].min()) / \
                              (obs["stochasticity"].max() - obs["stochasticity"].min())
        obs["current_state"] = (obs["current_state"] - obs["current_state"].min()) / \
                              (obs["current_state"].max() - obs["current_state"].min())

        return obs

    def _generate_stops(self):
        xy = np.random.rand(self.n_stops, 2) * self.max_box

        self.x = xy[:, 0]
        self.y = xy[:, 1]

        self.dist_matrix = distance_matrix(xy, xy)

        # Array with state definition, considering actual position and past positions
        self.memory = np.zeros((self.n_stops, self.n_stops))  # Allowed positions will be 1, otherwise 0
        # np.fill_diagonal(self.memory, 2)

    def _get_last_position(self):
        return self.stops[-1]

    def _get_xy(self, initial=False):
        state = self.stops[0] if initial else self._get_last_position()
        x = self.x[state]
        y = self.y[state]
        return x, y

    def _get_reward(self, st_pt, end_pt):
        if st_pt == end_pt:
            return -1 * self.max_box
        else:
            opt_reward = - np.random.normal(self.dist_matrix[st_pt, end_pt], self.traffic[st_pt, end_pt])
            target_reward = self.max_box

            if end_pt in self.stops:
                target_reward = -1 * self.max_box

            return self.weight_target * target_reward + self.weight_opt * opt_reward

    def reset(self):
        self.close()
        # Stops placeholder
        self.stops = []

        # Initialize distance matrix
        self._generate_stops()

        # Random variance values for all relative distances
        self._get_noise()

        # Random first stop
        first_stop = np.random.randint(self.n_stops)
        self.memory[:, first_stop] = 1

        self.stops.append(first_stop)

        # Reset variables
        self.window = None
        self.truncated = False
        self.terminated = False

        x = self.get_obs()
        first_stop_array = np.concatenate((x['distance_matrix'], x['current_state'], x['stochasticity'])).flatten()

        if self.render_mode == "human":
            self._render_frame()

        return first_stop_array, self.info

    def _available_actions(self):
        return list(set([i for i in range(self.n_stops)]) - set(self.stops))

    def get_action_mask(self):
        action_mask = np.ones(self.n_stops, dtype=np.float32)
        for stop in self.stops:
            action_mask[stop] = 0
        return action_mask

    def step(self, destination):
        # Get current state
        origin = self._get_last_position()

        # Get reward for such a move
        reward = self._get_reward(origin, destination)

        if destination not in self.stops: # TODO: what is doing with the memory
            self.memory[:, destination] += 1
            for i in self.stops:
                self.memory[:, i] += 1

            # Append new point to stops
            self.stops.append(destination)
        # else:
        #     self.truncated = True
        if len(self.stops) == self.n_stops:
            self.terminated = True

        x = self.get_obs()
        new_state = np.concatenate((x['distance_matrix'], x['current_state'], x['stochasticity'])).flatten()

        if self.render_mode == "human":
            self._render_frame()

        return new_state, reward, self.terminated, self.truncated, self.info

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))

        # Show stops
        for i in range(self.n_stops):
            pygame.draw.circle(canvas, "red", (self.grid_size * self.x[i], self.grid_size * self.y[i]), 5)

        # Show START
        if len(self.stops) > 0:
            xy = self._get_xy(initial=True)
            xytext = self.grid_size * (xy[0] + 0.1), self.grid_size * (xy[1] - 0.1)

            pygame.font.init()  # You have to call this at the start,
            # create a font object.
            # 1st parameter is the font file which is present in pygame.
            # 2nd parameter is size of the font
            font = pygame.font.Font('freesansbold.ttf', 12)
            text = font.render('START', False, (0, 0, 0))

            # Create a rectangular object for the text surface object
            textRect = text.get_rect()

            # Set the center of the rectangular object.
            textRect.center = (xytext[0], xytext[1])

            # Copying the text surface object to the display surface object at the center coordinate.
            canvas.blit(text, textRect)

        # Show itinerary
        if len(self.stops) > 1:
            pygame.draw.lines(canvas, "blue", False, [(self.grid_size * self.x[i], self.grid_size * self.y[i])
                                                      for i in self.stops], width=1)

            # Annotate END
            xy = self._get_xy(initial=False)
            xytext = self.grid_size * (xy[0] + 0.1), self.grid_size * (xy[1] - 0.1)

            text = font.render('END', False, (0, 0, 0))

            # Create a rectangular object for the text surface object
            textRect = text.get_rect()

            # Set the center of the rectangular object.
            textRect.center = (xytext[0], xytext[1])

            # Copying the text surface object to the display surface object at the center coordinate.
            canvas.blit(text, textRect)

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            # pygame.time.delay(1000)

            # We need to ensure that human-rendering occurs at the predefined frame rate.
            # The following line will automatically add a delay to keep the frame rate stable.
            self.clock.tick(self.metadata["render_fps"])

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
