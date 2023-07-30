# ===============================================================================================================================================
# This enviroment has been obtained from: https://github.com/TheoLvs/reinforcement-learning/blob/master/5.%20Delivery%20Optimization/delivery.py
# ===============================================================================================================================================

# Base Data Science snippet
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection
import gymnasium
from gymnasium import spaces
from sklearn.preprocessing import normalize

plt.style.use("seaborn-dark")


class DeliveryEnvironment(object):
    def __init__(self, n_stops=10, max_box=10, method="distance", **kwargs):

        # Initialization
        self.n_stops = n_stops
        self.action_space = self.n_stops
        self.observation_space = self.n_stops
        self.max_box = max_box
        self.stops = []
        self.method = method
        self.first_stop = None

        # Generate stops
        self._generate_constraints(**kwargs)
        self._generate_stops()
        self._generate_q_values()
        # self.render()

        # Initialize first point
        self.reset()

    def _generate_constraints(self, box_size=0.2, traffic_intensity=5):

        if self.method == "traffic_box":
            x_left = np.random.rand() * (self.max_box) * (1 - box_size)
            y_bottom = np.random.rand() * (self.max_box) * (1 - box_size)

            x_right = x_left + np.random.rand() * box_size * self.max_box
            y_top = y_bottom + np.random.rand() * box_size * self.max_box

            self.box = (x_left, x_right, y_bottom, y_top)
            self.traffic_intensity = traffic_intensity

    def _generate_stops(self):

        if self.method == "traffic_box":

            points = []
            while len(points) < self.n_stops:
                x, y = np.random.rand(2) * self.max_box
                if not self._is_in_box(x, y, self.box):
                    points.append((x, y))

            xy = np.array(points)

        else:
            # Generate geographical coordinates
            xy = np.random.rand(self.n_stops, 2) * self.max_box

        self.x = xy[:, 0]
        self.y = xy[:, 1]

    def _generate_q_values(self, box_size=0.2):

        # Generate actual Q Values corresponding to time elapsed between two points
        if self.method in ["distance", "traffic_box"]:
            xy = np.column_stack([self.x, self.y])
            self.q_stops = cdist(xy, xy)
        elif self.method == "time":
            self.q_stops = np.random.rand(self.n_stops, self.n_stops) * self.max_box
            np.fill_diagonal(self.q_stops, 0)
        else:
            raise Exception("Method not recognized")

    def render(self, return_img=False):

        fig = plt.figure(figsize=(7, 7))
        ax = fig.add_subplot(111)
        ax.set_title("Delivery Stops")

        # Show stops
        ax.scatter(self.x, self.y, c="red", s=50)

        # Show START
        if len(self.stops) > 0:
            xy = self._get_xy(initial=True)
            xytext = xy[0] + 0.1, xy[1] - 0.05
            ax.annotate("START", xy=xy, xytext=xytext, weight="bold")

        # Show itinerary
        if len(self.stops) > 1:
            ax.plot(self.x[self.stops], self.y[self.stops], c="blue", linewidth=1, linestyle="--")

            # Annotate END
            xy = self._get_xy(initial=False)
            xytext = xy[0] + 0.1, xy[1] - 0.05
            ax.annotate("END", xy=xy, xytext=xytext, weight="bold")

        if hasattr(self, "box"):
            left, bottom = self.box[0], self.box[2]
            width = self.box[1] - self.box[0]
            height = self.box[3] - self.box[2]
            rect = Rectangle((left, bottom), width, height)
            collection = PatchCollection([rect], facecolor="red", alpha=0.2)
            ax.add_collection(collection)

        plt.xticks([])
        plt.yticks([])

        if return_img:
            # From https://ndres.me/post/matplotlib-animated-gifs-easily/
            fig.canvas.draw_idle()
            image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
            image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            plt.close()
            return image
        else:
            plt.show()

    def reset(self):

        # Stops placeholder
        self.stops = []

        # Random first stop
        # self.first_stop = np.random.randint(self.n_stops)
        self.first_stop = 0
        self.stops.append(self.first_stop)

        return self.first_stop

    def step(self, destination):

        # Get current state
        state = self._get_state()
        new_state = destination

        # Get reward for such a move
        reward = self._get_reward(state, new_state)

        # Append new_state to stops
        self.stops.append(destination)
        done = len(self.stops) == self.n_stops

        return new_state, -reward, done

    def _get_state(self):
        return self.stops[-1]

    def _get_xy(self, initial=False):
        state = self.stops[0] if initial else self._get_state()
        x = self.x[state]
        y = self.y[state]
        return x, y

    def _get_reward(self, state, new_state):
        base_reward = self.q_stops[state, new_state]

        if self.method == "distance":
            return base_reward
        elif self.method == "time":
            return base_reward + np.random.randn()
        elif self.method == "traffic_box":

            # Additional reward correspond to slowing down in traffic
            xs, ys = self.x[state], self.y[state]
            xe, ye = self.x[new_state], self.y[new_state]
            intersections = self._calculate_box_intersection(xs, xe, ys, ye, self.box)
            if len(intersections) > 0:
                i1, i2 = intersections
                distance_traffic = np.sqrt((i2[1] - i1[1]) ** 2 + (i2[0] - i1[0]) ** 2)
                additional_reward = distance_traffic * self.traffic_intensity * np.random.rand()
            else:
                additional_reward = np.random.rand()

            return base_reward + additional_reward

    @staticmethod
    def _calculate_point(x1, x2, y1, y2, x=None, y=None):

        if y1 == y2:
            return y1
        elif x1 == x2:
            return x1
        else:
            a = (y2 - y1) / (x2 - x1)
            b = y2 - a * x2

            if x is None:
                x = (y - b) / a
                return x
            elif y is None:
                y = a * x + b
                return y
            else:
                raise Exception("Provide x or y")

    def _is_in_box(self, x, y, box):
        # Get box coordinates
        x_left, x_right, y_bottom, y_top = box
        return x >= x_left and x <= x_right and y >= y_bottom and y <= y_top

    def _calculate_box_intersection(self, x1, x2, y1, y2, box):

        # Get box coordinates
        x_left, x_right, y_bottom, y_top = box

        # Intersections
        intersections = []

        # Top intersection
        i_top = self._calculate_point(x1, x2, y1, y2, y=y_top)
        if i_top > x_left and i_top < x_right:
            intersections.append((i_top, y_top))

        # Bottom intersection
        i_bottom = self._calculate_point(x1, x2, y1, y2, y=y_bottom)
        if i_bottom > x_left and i_bottom < x_right:
            intersections.append((i_bottom, y_bottom))

        # Left intersection
        i_left = self._calculate_point(x1, x2, y1, y2, x=x_left)
        if i_left > y_bottom and i_left < y_top:
            intersections.append((x_left, i_left))

        # Right intersection
        i_right = self._calculate_point(x1, x2, y1, y2, x=x_right)
        if i_right > y_bottom and i_right < y_top:
            intersections.append((x_right, i_right))

        return intersections


class DeliveryEnv(gymnasium.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, n_stops=10, max_box=10, method="distance", flag_action_mask=False, **kwargs):
        self.n_stops = n_stops
        self.max_box = max_box
        self.method = method
        self.kwargs = kwargs
        self.delivery_env = DeliveryEnvironment(n_stops=n_stops, max_box=max_box, method=method, **kwargs)
        self.action_space = spaces.Discrete(n_stops)
        self.observation_space = spaces.Box(low=0, high=1, shape=(n_stops * 4,), dtype=np.float32)
        # self.observation_space = spaces.Dict({
        #    "action_mask": spaces.Box(0, 1, shape=(n_stops,), dtype=np.float32),
        #    "real_obs": spaces.Box(low=0, high=1, shape=(n_stops * 4,), dtype=np.float32)
        # })
        self.state = None
        self.truncated = False
        self.terminated = False
        self.reward_episode = 0
        self.flag_action_mask = flag_action_mask
        self.info = None

    def step(self, action):
        if action in self.delivery_env.stops:
            reward = -100
            state = self.state
            self.truncated = True
        else:
            state, reward, done = self.delivery_env.step(action)
        if len(self.delivery_env.stops) == self.n_stops:
            self.terminated = True

        self.state = state
        one_hot_state = self.one_hot_encode(state, self.n_stops)

        # Encode and normalize the positon of all the points
        points_normalized = np.concatenate(
            (normalize([self.delivery_env.x]), normalize([self.delivery_env.y]))).flatten()

        # Accumulated reward of the episode
        self.reward_episode += reward

        if self.flag_action_mask:
            self.info = self.get_action_mask()

        return np.concatenate((one_hot_state, points_normalized,
                               self.stops_encoder())).flatten(), reward, self.terminated, self.truncated, self.info
        # return {
        #           "real_obs": np.concatenate((one_hot_state, points_normalized, self.stops_encoder())).flatten(),
        #           "action_mask": self.get_action_mask()
        #       }, reward, self.terminated, self.truncated, {}

    def get_action_mask(self):
        action_mask = np.ones(self.n_stops, dtype=np.float32)
        for stop in self.delivery_env.stops:
            action_mask[stop] = 0
        return action_mask

    def reset(self, seed=None, options=None):
        self.delivery_env = DeliveryEnvironment(n_stops=self.n_stops, max_box=self.max_box, method=self.method,
                                                **self.kwargs)
        self.state = self.delivery_env.reset()
        one_hot_state = self.one_hot_encode(self.state, self.n_stops)

        # Encode and normalize the positon of all the points
        points_normalized = np.concatenate(
            (normalize([self.delivery_env.x]), normalize([self.delivery_env.y]))).flatten()

        # Reset attributes
        self.reward_episode = 0
        self.truncated = False
        self.terminated = False

        if self.flag_action_mask:
            self.info = self.get_action_mask()

        return np.concatenate((one_hot_state, points_normalized, self.stops_encoder())).flatten(), self.info
        # return {
        #           "real_obs": np.concatenate((one_hot_state, points_normalized, self.stops_encoder())).flatten(),
        #           "action_mask": self.get_action_mask()
        #       }, {}

    def one_hot_encode(self, state, n_stops):
        one_hot = np.zeros(n_stops)
        one_hot[state] = 1
        return one_hot

    def stops_encoder(self):
        result = [1 if i in self.delivery_env.stops else 0 for i in range(self.n_stops)]
        return result

    def render(self, mode='human', close=False):
        return self.delivery_env.render(return_img=True)


class DeliveryEnv_v3(gymnasium.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self,
                 n_stops=10,
                 max_box=10,
                 weight_target=1,
                 weight_opt=1,
                 method="distance",
                 flag_action_mask=False,
                 **kwargs):

        self.n_stops = n_stops
        self.max_box = max_box
        self.method = method
        self.kwargs = kwargs
        self.delivery_env = DeliveryEnvironment(n_stops=n_stops, max_box=max_box, method=method, **kwargs)
        self.action_space = spaces.Discrete(n_stops)
        self.observation_space = spaces.Box(low=0, high=1, shape=(n_stops * 4,), dtype=np.float32)
        self.state = None
        self.truncated = False
        self.terminated = False
        self.reward_episode = 0
        self.flag_action_mask = flag_action_mask
        self.info = {}
        self.weight_target = weight_target
        self.weight_opt = weight_opt

    def standard_reward(self, reward, action):

        # remove current stop from the states
        previous_stops = self.delivery_env.stops[:-1]

        # if the action selected is to remain in the same spot
        if action == previous_stops[-1]:
            return -1 * self.max_box
        else:
            opt_reward = reward
            target_reward = self.max_box  # TODO: Add traffic (Quique´s version)

            if action in previous_stops:
                target_reward = -1 * self.max_box

            return self.weight_target * target_reward + self.weight_opt * opt_reward

    def step(self, action):

        state, reward, self.truncated = self.delivery_env.step(action)

        reward = self.standard_reward(reward, action)

        self.state = state
        one_hot_state = self.one_hot_encode(state, self.n_stops)

        # Encode and normalize the positon of all the points
        points_normalized = np.concatenate(
            (normalize([self.delivery_env.x]), normalize([self.delivery_env.y]))).flatten()

        # Accumulated reward of the episode
        self.reward_episode += reward

        if self.flag_action_mask:
            self.info['mask'] = self.get_action_mask()

        return np.concatenate((one_hot_state, points_normalized,
                               self.stops_encoder())).flatten(), reward, self.terminated, self.truncated, self.info

    def get_action_mask(self):
        action_mask = np.ones(self.n_stops, dtype=np.float32)
        for stop in self.delivery_env.stops:
            action_mask[stop] = 0
        return action_mask

    def reset(self, seed=None, options=None):
        # self.delivery_env = DeliveryEnvironment(n_stops=self.n_stops, max_box=self.max_box, method=self.method,
        #                                         **self.kwargs)
        self.state = self.delivery_env.reset()
        one_hot_state = self.one_hot_encode(self.state, self.n_stops)

        # Encode and normalize the positon of all the points
        points_normalized = np.concatenate(
            (normalize([self.delivery_env.x]), normalize([self.delivery_env.y]))).flatten()

        # Reset attributes
        self.reward_episode = 0
        self.truncated = False
        self.terminated = False

        if self.flag_action_mask:
            self.info['mask'] = self.get_action_mask()

        return np.concatenate((one_hot_state, points_normalized, self.stops_encoder())).flatten(), self.info

    def one_hot_encode(self, state, n_stops):
        one_hot = np.zeros(n_stops)
        one_hot[state] = 1
        return one_hot

    def stops_encoder(self):
        result = [1 if i in self.delivery_env.stops else 0 for i in range(self.n_stops)]
        return result

    def render(self, mode='human', close=False):
        return self.delivery_env.render(return_img=True)


class DeliveryEnv_structure2vec(gymnasium.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self,
                 n_stops=10,
                 max_box=10,
                 weight_target=1,
                 weight_opt=1,
                 method="distance",
                 flag_action_mask=False,
                 **kwargs):

        self.n_stops = n_stops
        self.max_box = max_box
        self.method = method
        self.kwargs = kwargs
        self.delivery_env = DeliveryEnvironment(n_stops=n_stops, max_box=max_box, method=method, **kwargs)
        self.action_space = spaces.Discrete(n_stops)
        self.observation_space = spaces.Box(low=0, high=1, shape=(n_stops * 4,), dtype=np.float32)
        self.state = None
        self.truncated = False
        self.terminated = False
        self.reward_episode = 0
        self.flag_action_mask = flag_action_mask
        self.info = {}
        self.weight_target = weight_target
        self.weight_opt = weight_opt

        x = self.delivery_env.x.reshape(-1, 1)
        y = self.delivery_env.y.reshape(-1, 1)
        self.dist_matrix = np.sqrt((x - x.T)**2 + (y - y.T)**2)

    def standard_reward(self, reward, action):

        # remove current stop from the states
        previous_stops = self.delivery_env.stops[:-1]

        # if the action selected is to remain in the same spot
        if action == previous_stops[-1]:
            return -1 * self.max_box
        else:
            opt_reward = reward
            target_reward = self.max_box

            if action in previous_stops:
                target_reward = -1 * self.max_box

            return self.weight_target * target_reward + self.weight_opt * opt_reward

    def step(self, action):

        state, reward, self.truncated = self.delivery_env.step(action)

        reward = self.standard_reward(reward, action)

        self.state = state
        one_hot_state = self.one_hot_encode(state, self.n_stops)

        # Encode and normalize the positon of all the points
        points_normalized = np.concatenate(
            (normalize([self.delivery_env.x]), normalize([self.delivery_env.y]))).flatten()

        # Accumulated reward of the episode
        self.reward_episode += reward

        if self.flag_action_mask:
            self.info['mask'] = self.get_action_mask()

        return one_hot_state, reward, self.terminated, self.truncated, self.info

    def get_action_mask(self):
        action_mask = np.ones(self.n_stops, dtype=np.float32)
        for stop in self.delivery_env.stops:
            action_mask[stop] = 0
        return action_mask

    def reset(self, seed=None, options=None):
        # self.delivery_env = DeliveryEnvironment(n_stops=self.n_stops, max_box=self.max_box, method=self.method,
        #                                         **self.kwargs)
        self.state = self.delivery_env.reset()
        one_hot_state = self.one_hot_encode(self.state, self.n_stops)

        # Encode and normalize the positon of all the points
        points_normalized = np.concatenate(
            (normalize([self.delivery_env.x]), normalize([self.delivery_env.y]))).flatten()

        # Reset attributes
        self.reward_episode = 0
        self.truncated = False
        self.terminated = False

        if self.flag_action_mask:
            self.info['mask'] = self.get_action_mask()

        return one_hot_state, self.info

    def one_hot_encode(self, state, n_stops):
        one_hot = np.zeros(n_stops)
        one_hot[state] = 1
        return one_hot

    def stops_encoder(self):
        result = [1 if i in self.delivery_env.stops else 0 for i in range(self.n_stops)]
        return result

    def render(self, mode='human', close=False):
        return self.delivery_env.render(return_img=True)

