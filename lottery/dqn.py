import math

import numpy as np
from keras.activations import relu
from keras.layers import (Activation, Concatenate, Dense, Flatten, Input,
                          InputLayer, Lambda, Dropout)
from keras.models import Model, Sequential, load_model
from keras.optimizers import Adam, RMSprop, Nadam
from keras.backend import categorical_crossentropy
from tensorflow import nn

from environment import LotteryEnv

MAX_EPSILON = 1.0       # 1.0
MIN_EPSILON = 0.01      # 0.01
LAMBDA = 0.00005         # 0.0005
GAMMA = 0.95            # 0.95
BATCH_SIZE = 32         # 32
TAU = 0.08               # 0.08
RANDOM_REWARD_STD = 1.0  # 1.0
LEARNING_RATE = 0.0002   # 0.0003


def relu6(x): return nn.relu6(x)


def network_builder(binary_ranks=10, inputs_number=69, activation='linear',
                    dropout=.0):
    inputs = []

    for _ in range(inputs_number):
        inputs.append(Input(shape=(binary_ranks,)))
    merged = Concatenate(axis=1)(inputs)
    dense0 = Dense(units=2048, activation=relu6, use_bias=True)(merged)
    dense1 = Dense(units=1024, activation=relu6, use_bias=True)(
        Dropout(dropout)(dense0) if dropout else dense0)
    dense2 = Dense(units=512, activation=relu6, use_bias=True)(
        Dropout(dropout)(dense1) if dropout else dense1)
    # drop5 = Dropout(0.5)(dense5)
    # dense4_l = Lambda(lambda x: x[:,::2])(dense4)
    # dense4_r = Lambda(lambda x: x[:,1::2])(dense4)
    # dense5 = Dense(units=256, activation=relu6, use_bias=True)(dense4_l)
    # dense6 = Dense(units=256, activation=relu6, use_bias=True)(dense4_r)
    # merged_1 = Concatenate(axis=1)([dense5, dense6])
    dense3 = Dense(units=256, activation=relu6, use_bias=True)(
        Dropout(dropout)(dense2) if dropout else dense2)
    output = Dense(units=inputs_number, activation=activation,  # softmax
                   use_bias=True)(dense3)

    return Model(inputs=inputs, outputs=[output])


def to_binary(a, binary_ranks):
    return ((int(a) & (1 << np.arange(binary_ranks))) > 0)[::-1].astype(int)


def to_binary_list(l, binary_ranks):
    return [to_binary(el, binary_ranks).reshape(1, -1) for el in l]


class Memory:
    def __init__(self, max_memory, state_size, state_number):
        self._max_memory = max_memory
        self._state_size = state_size
        self._state_number = state_number
        self._samples = []

    def add_sample(self, sample):
        if (sample[3] is None):
            sample = (sample[0], sample[1], sample[2],
                      [np.zeros(self._state_size).reshape(1, -1) for _ in range(self._state_number)])
        self._samples.append(sample)
        if len(self._samples) > self._max_memory:
            self._samples.pop(0)

    def sample(self, no_samples):
        def take_randoms(n):
            np.random.shuffle(self._samples)
            return self._samples[-n:]

        if no_samples > len(self._samples):
            return take_randoms(len(self._samples))
        else:
            return take_randoms(no_samples)

    @property
    def num_samples(self):
        return len(self._samples)


class LottoNN:
    def __init__(self, name, path=None, steps=1000, data=None, metadata=None, binary_ranks=10,
                 balls_number=69, actions_number=69, memory_size=10000, single=False, optimizer="adam"):
        self.name = name
        self.binary_ranks = binary_ranks
        self.balls_number = balls_number
        self.actions_number = actions_number
        self.steps = steps - 1
        self.data = data
        self.metadata = metadata
        self.env = LotteryEnv(self.steps, self.data, self.metadata,
                              self.binary_ranks, self.balls_number, self.actions_number)
        self.arr = np.identity(balls_number)
        self.memory = Memory(memory_size, binary_ranks, balls_number)

        if optimizer == "adam":
            optimizer = Adam(learning_rate=LEARNING_RATE)
        elif optimizer == "nadam":
            optimizer = Nadam(learning_rate=LEARNING_RATE)
        elif optimizer == "rmsprop":
            optimizer = RMSprop(learning_rate=LEARNING_RATE)

        if path is not None:
            self.primary_network = load_model(
                path, custom_objects={'relu6': relu6})
        else:
            self.primary_network = network_builder(binary_ranks, balls_number)
            self.primary_network.compile(
                loss='mse', optimizer=optimizer, metrics=['mae'])
            # loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

        if single:
            self.target_network = None
        else:
            self.target_network = network_builder(binary_ranks, balls_number)

        self.primary_network.summary()

    def choose_actions(self, state, eps):
        actions = [0] * self.actions_number

        if np.random.random() < eps:
            p = np.random.rand(1, self.balls_number)
        else:
            p = self.primary_network.predict(state)

        for i in range(self.actions_number):
            actions[i] = np.argmax(p)
            p[0][actions[i]] = -10. ** 10

        return actions

    def train(self):
        if self.memory.num_samples < BATCH_SIZE * 3:
            return .0, .0
        batch = self.memory.sample(BATCH_SIZE)
        states, actions, rewards, next_states = [
            np.array(el) for el in zip(*batch)]
        states = [np.array(el).reshape(-1, self.binary_ranks)
                  for el in zip(*states)]
        next_states = [np.array(el).reshape(-1, self.binary_ranks)
                       for el in zip(*next_states)]

        prim_qt = self.primary_network.predict(states, batch_size=BATCH_SIZE)
        prim_qtp1 = self.primary_network.predict(
            next_states, batch_size=BATCH_SIZE)
        target_q = prim_qt
        updates = rewards
        valid_idxs = np.array([el.sum(axis=1)
                               for el in next_states]).sum(axis=0) != 0
        batch_idxs = np.arange(BATCH_SIZE)

        if self.target_network is None:
            updates[valid_idxs] += GAMMA * \
                np.amax(prim_qtp1[valid_idxs, :], axis=1)
        else:
            prim_action_tp1 = np.argmax(prim_qtp1, axis=1)
            q_from_target = self.target_network.predict(
                next_states, batch_size=BATCH_SIZE)
            updates[valid_idxs] += GAMMA * q_from_target[batch_idxs[valid_idxs],
                                                         prim_action_tp1[valid_idxs]]

        target_q[batch_idxs, np.array(actions)[:, :1].reshape(-1)] = updates
        loss = self.primary_network.train_on_batch(states, target_q)
        if self.target_network is not None:
            for t, e in zip(self.target_network.trainable_weights, self.primary_network.trainable_weights):
                t.assign(t * (1.0 - TAU) + e * TAU)

        return loss

    def fit(self, num_episodes=20):
        eps = MAX_EPSILON
        steps = 0
        self.r_avg_list = []
        for i in range(num_episodes):
            cnt = 0
            avg_loss = .0
            r_sum = 0
            done = False
            state = self.env.reset()
            state = to_binary_list(state, self.binary_ranks)
            while not done:
                actions = self.choose_actions(state, eps)
                next_state, reward, done, _ = self.env.step(actions)
                next_state = to_binary_list(next_state, self.binary_ranks)
                if done:
                    next_state = None
                self.memory.add_sample((state, actions, reward, next_state))
                loss, _ = self.train()
                avg_loss = np.add(avg_loss, loss)
                state = next_state
                steps += 1
                cnt += 1
                eps = MIN_EPSILON + (MAX_EPSILON - MIN_EPSILON) * \
                    math.exp(-LAMBDA * steps)
                if done:
                    avg_loss = np.divide(avg_loss, cnt)
                    print(
                        f"Episode: {i}, Reward: {r_sum}, avg loss: {avg_loss:.3f}, eps: {eps:.3f}")

                r_sum += reward
            self.r_avg_list.append(r_sum / 10)

        self.primary_network.save("{}.h5".format(self.name))

        return self.r_avg_list

    def rescue(self):
        self.primary_network.save("{}.h5".format(self.name))

        return self.r_avg_list

    def _predict(self, sum):
        state = [to_binary(int(el / self.steps * 1000),
                           self.binary_ranks).reshape(1, -1) for el in sum]

        prediction = []
        p = self.primary_network.predict(state)
        for _ in range(self.balls_number):
            prediction.append(np.argmax(p))
            p[0][prediction[-1]] = -10. ** 10

        return list(map(lambda x: x + 1, prediction))

    def predict(self, sequentially=False):
        sum = [0] * self.balls_number

        if sequentially:
            predictions = []

        for i in range(self.steps):
            for b in self.data[i]:
                sum[b] += 1.0
            if sequentially:
                predictions.append(self._predict(sum))

        if sequentially:
            return predictions
        else:
            return self._predict(sum)
