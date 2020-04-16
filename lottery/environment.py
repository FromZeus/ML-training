import secrets

import gym
import numpy as np
from gym import spaces
from gym.utils import seeding


def fast(x, n):
    return np.multiply(np.divide(x, np.add(n, 1)), 1000)


vfast = np.vectorize(fast)


class LotteryEnv(gym.Env):
    def __init__(self, steps, data=None, metadata=None, n=10, balls_number=69, actions_number=5, draw_length=5, seed=None):
        self.seed(seed)
        self.steps = steps
        self.data = data
        self.metadata = metadata
        self.balls_number = balls_number
        self.draw_length = draw_length
        self.random = secrets.SystemRandom(seed)

        spaces_list = []
        for _ in range(balls_number):
            spaces_list.append(spaces.Discrete(n))
        self.observation_space = spaces.Tuple(spaces_list)
        spaces_list = []
        for _ in range(actions_number):
            spaces_list.append(spaces.Discrete(balls_number))
        self.action_space = spaces.Tuple(spaces_list)
        self.reset()

    def round(self):
        draw = np.arange(self.balls_number)
        np.random.shuffle(draw)
        draw = draw[:self.draw_length]

        return draw

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, actions):
        assert self.action_space.contains(actions)
        self.steps_counter += 1
        done = False
        reward = 0.

        balls = self.data[self.steps_counter] if self.data else self.round()
        if self.metadata:
            indexes = self.metadata[self.steps_counter - 1]

        for idx, action in enumerate(actions):
            if self.metadata:
                if action in indexes:
                    reward += 10
            else:
                if action in balls:
                    reward += 10. / (idx + 1)
        if self.steps_counter == self.steps:
            done = True

        self.set_state(self.steps_counter, balls)

        return self.state, reward, done, {}

    def set_state(self, n, balls):
        for el in balls:
            self.sum[el] += 1.0
        # self.state = vfast(self.state, n)
        for i in range(len(self.state)):
            self.state[i] = int(self.sum[i] / (n + 1) * 1000)

    def reset(self):
        self.steps_counter = 0
        self.sum = [0] * self.balls_number
        self.state = [0] * self.balls_number
        if self.data is not None:
            # self.np_random.shuffle(self.data)
            balls = self.data[0]
        else:
            balls = self.round()
        self.set_state(0, balls)

        return self.state
