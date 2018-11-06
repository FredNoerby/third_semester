import numpy as np
import random


class EnvironmentB:

    def __init__(self):
        self.banana_pose = 0
        self.reward_dict = {'move': - 0.001, 'illegal': - 0.1, 'guess_pos': 1, 'guess_neg': -1}
        self.sensor_test = [[np.array([20, 20, 100, 40]), np.array([20, 20, 40, 100])],
                            [np.array([20, 20, 40, 100]), np.array([20, 20, 100, 40])]]
        self.move_dict = {}
        self.observation_space = np.array([0, 0, 0, 0, 0])
        self.action_space = ActionSpace(4)


    def _sense(self):
        for i in range(1, 5):
            self.observation_space[i] = self.sensor_test[self.observation_space[0]][self.banana_pose][i-1] \
                                        # + random.randint(-1, 1)
        print("Sense = {}".format(self.observation_space))


    def _move(self, action):
        if action == 0:
            self.observation_space[0] = 0
        else:
            self.observation_space[0] = 1

    def reset(self):
        self.banana_pose = random.randint(0, 1)
        # print("Banana pose is: {}".format(self.banana_pose))
        n = random.randint(0, 1)
        self.observation_space[0] = n
        self._sense()

        # print("Cameraposition is: {}".format(self.observation_space[0]))

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
            if self.banana_pose == 0:
                reward = self.reward_dict['guess_pos']
            else:
                reward = self.reward_dict['guess_neg']
            done = True

        elif action == 3:
            if self.banana_pose == 1:
                reward = self.reward_dict['guess_pos']
            else:
                reward = self.reward_dict['guess_neg']
            done = True

        else:
            reward = self.reward_dict['illegal']

        info = {}
        print("Action: {} Observation: {} bananpose: {} reward {} ".format(action, self.observation_space, self.banana_pose, reward))
        return self.observation_space, reward, done, info

    def render(self, mode='human'):
        # Placeholder for rendering
        print(mode)



class ActionSpace:
    def __init__(self, n):
        self.n = n

    def sample(self):

        return rand_sample
