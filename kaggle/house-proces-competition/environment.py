import secrets

import gym
import numpy as np
from gym import spaces
from gym.utils import seeding


class HouseEnv(gym.Env):
    def __init__(self, X, y, seed=None):
        self.seed(seed)
        self.X = X
        self.y = y
        self.random = secrets.SystemRandom(seed)

        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        done = False
        reward = 0.

        price = self.y[self.steps_counter]
        self.steps_counter += 1

        reward = 1. / abs(price - action)
        if self.steps_counter == len(self.y) - 1:
            done = True
        self.state = self.X.iloc[[self.steps_counter]]

        return self.state, reward, done, {}

    def reset(self):
        self.steps_counter = 0
        self.state = [0] * len(self.X.columns)

        return self.state
