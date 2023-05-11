import os
import numpy as np
import torch
from collections import deque
import random
import copy
import datetime

# Imports inside the library
from ..config import SAVE_MODEL

class DQNAgent:
    def __init__(self, env, model, lr=1e-3, gamma=0.99, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995, buffer_size=100000, batch_size=64, update_target_freq=1000):
        self.env = env
        self.model = model
        self.target_model = copy.deepcopy(model)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.buffer = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.update_target_freq = update_target_freq
        self.steps = 0

    def act(self, state):
        # Act is implementing e-greedy selector
        if np.random.rand() < self.epsilon:
            return self.env.action_space.sample()
        else:
            state = torch.tensor(state, dtype=torch.float32)
            q_values = self.model(state)
            return torch.argmax(q_values).item()

    def remember(self, state, action, reward, next_state, terminated, truncated):
        self.buffer.append((state, action, reward, next_state, terminated, truncated))

    def learn(self):
        if len(self.buffer) < self.batch_size:
            return

        minibatch = random.sample(self.buffer, self.batch_size)

        states, actions, rewards, next_states, terminateds, truncateds = zip(*minibatch)
        states = torch.tensor(np.array(states), dtype=torch.float32)
        actions = torch.tensor(np.array(actions), dtype=torch.int64)
        rewards = torch.tensor(np.array(rewards), dtype=torch.float32)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32)
        terminateds = torch.tensor(np.array(terminateds), dtype=torch.uint8)
        truncateds = torch.tensor(np.array(truncateds), dtype=torch.uint8)
        # TODO: For the Cart Pole terminated and truncated seems to be doing the same. Review for other environments
        done = torch.logical_or(terminateds, truncateds).int()

        q_values = self.model(states).gather(1, actions.unsqueeze(1)).squeeze()
        next_q_values = self.target_model(next_states).max(1)[0]
        target_q_values = rewards + (1 - done) * self.gamma * next_q_values

        ### -------------- TECHNICAL EXPLANATION: Detach target value --------------------------------------------------
        # Even though loss.backward() computes gradients for all parameters that require gradients, including those in the
        # target network if you don't detach target_q_values, the optimizer won't update the target network's parameters '
        # because they weren't included when the optimizer was created.
        # However, it's still better practice to detach target_q_values for a couple of reasons:
        #
        #   - It makes it clear that the target network's parameters are not part of the computational graph for the loss,
        #     which helps avoid confusion.
        #   - It saves computation, because loss.backward() won't need to compute unnecessary
        #     gradients for the target network's parameters.
        ### -------------- END TECHNICAL EXPLANATION: Detach target value ----------------------------------------------
        loss = torch.nn.functional.mse_loss(q_values, target_q_values.detach())

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update epsilon
        self.steps += 1
        self.epsilon = max(self.epsilon_end, self.epsilon_decay * self.epsilon) # TODO: It is updating the epsilon each step: 64 batch size

        # Update target model
        if self.steps % self.update_target_freq == 0:
            self.target_model.load_state_dict(self.model.state_dict())

    def save(self, path=SAVE_MODEL):
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        filename = f"dqn_agent_{timestamp}.pt"
        filepath = path + filename

        # Create directory if it doesn't exist
        os.makedirs(path, exist_ok=True)

        torch.save(self.model.state_dict(), filepath)
        print(f"Checkpoint saved in {filepath}")

        return filepath

    def load(self, path):
        self.model.load_state_dict(torch.load(path))
        self.target_model.load_state_dict(self.model.state_dict())
