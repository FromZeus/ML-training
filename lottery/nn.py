import numpy as np
from keras.layers import (Activation, Concatenate, Dense, Flatten, Input,
                          InputLayer)
from keras.models import Model, Sequential, load_model
from keras.optimizers import Adam

from environment import LotteryEnv


class NeuranNetwork:
    def __init__(self, name, path=None, steps=1000, data=None, n=10, balls_number=69, actions_number=5):
        self.name = name
        self.n = n
        self.balls_number = balls_number
        self.actions_number = actions_number
        self.env = LotteryEnv(steps, data)
        self.arr = np.identity(balls_number)

        if path is not None:
            self.model = load_model(path)
        else:
            balls = []
            for _ in range(balls_number):
                balls.append(Input(shape=(n,)))
            merged = Concatenate(axis=1)(balls)
            dense0 = Dense(units=4096, activation='relu',
                           use_bias=True)(merged)
            dense1 = Dense(units=2048, activation='relu',
                           use_bias=True)(dense0)
            dense2 = Dense(units=1024, activation='relu',
                           use_bias=True)(dense1)
            dense3 = Dense(units=512, activation='relu', use_bias=True)(dense2)
            dense4 = Dense(units=256, activation='relu', use_bias=True)(dense3)
            dense5 = Dense(units=128, activation='relu', use_bias=True)(dense4)
            ball_output = Dense(units=balls_number, activation='softmax',  # linear
                                use_bias=True)(dense5)
            self.model = Model(inputs=balls, outputs=[ball_output])
            self.model.compile(loss='mse', optimizer='adam', metrics=['mae'])
            # self.model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
        self.model.summary()

    def to_binary(self, a):
        return ((int(a) & (1 << np.arange(self.n))) > 0)[::-1].astype(int)

    def fit(self, num_episodes=20):
        eps = 0.5
        y = 0.95
        decay_factor = 0.999
        r_avg_list = []
        for i in range(num_episodes):
            s = self.env.reset()
            eps *= decay_factor
            if i % 10 == 0:
                print("Episode {} of {}".format(i + 1, num_episodes))
            done = False
            r_sum = 0
            while not done:
                _s = [0] * self.balls_number
                for i in range(self.balls_number):
                    _s[i] = self.to_binary(s[i]).reshape(-1, self.n)
                actions = [0] * self.actions_number
                if np.random.random() < eps:
                    for i in range(self.actions_number):
                        # a = np.random.randint(0, self.balls_number)
                        actions[i] = np.random.randint(0, self.balls_number)
                else:
                    p = self.model.predict(_s)
                    for _ in range(self.actions_number):
                        p[0][np.argmax(p)] = .0
                    for i in range(self.actions_number):
                        # a = np.argmax(p)
                        actions[i] = np.argmax(p)
                        p[0][actions[i]] = .0

                new_s, r, done, _ = self.env.step(actions)
                _new_s = [0] * self.balls_number
                for i in range(self.balls_number):
                    _new_s[i] = self.to_binary(s[i]).reshape(-1, self.n)
                target = r + y * np.max(self.model.predict(_new_s))
                target_vec = self.model.predict(_s)[0]
                target_vec[actions] = target
                self.model.fit(_s, target_vec.reshape(-1, 69),
                               epochs=1, verbose=0)
                s = new_s
                r_sum += r
            r_avg_list.append(r_sum / 10)

        self.model.save("{}.h5".format(self.name))

        return r_avg_list

    def predict(self):
        pass
