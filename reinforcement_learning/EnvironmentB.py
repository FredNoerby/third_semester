import numpy as np
import random


class EnvironmentB:

    def __init__(self):
        self.banana_pose = 1
        self.reward_dict = {'move': -1, 'illegal': -5, 'guess_pos': 10, 'guess_neg': -10}
        self.sensor_test = {0: np.array([0, 20, 20, 40, 40]), 1: np.array([1, 20, 20, 40, 41])}
        self.move_dict = {}
        self.observation_space = np.array([0, 0, 0, 0, 0])
        self.action_space = ActionSpace(4)


    def _sense(self):
        for i in range(1,5):
            self.observation_space[i] = self.observation_space[i] + random.randint(-10, 10)


    def _move(self, action):
        if action == 0:
            self.observation_space = self.sensor_test[0]
        else:
            self.observation_space = self.sensor_test[1]

    def reset(self):
        n = random.randint(0, 1)
        if n == 0:
            self.observation_space = self.sensor_test[0]
            self._sense()

        else:
            self.observation_space = self.sensor_test[1]
            self._sense()

        if random.randint(1, 2) == 2:
            self.banana_pose = 2

        return self.observation_space

    def step(self, action):
        done = False
        if action == 0 and self.observation_space[0] != 0:
            self._move(action)
            self._sense()
            reward = self.reward_dict['move']

        elif action == 1 and self.observation_space[0] != 1:
            self._move(action)
            self._sense()
            reward = self.reward_dict['move']

        elif action == 2:
            done = True
            if self.banana_pose == 1:
                reward = self.reward_dict['guess_pos']
            else:
                reward = self.reward_dict['guess_neg']
        elif action == 3:
            done = True
            if self.banana_pose == 2:
                reward = self.reward_dict['guess_pos']
            else:
                reward = self.reward_dict['guess_neg']
        else:
            reward = self.reward_dict['illegal']

        info = {}

        return self.observation_space, reward, done, info

    def render(self, mode='human'):
        # Placeholder for rendering
        print(mode)


class ActionSpace:
    def __init__(self, n):
        self.n = n

    def sample(self):

        return rand_sample

