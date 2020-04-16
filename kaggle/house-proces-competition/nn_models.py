import math
import secrets
import sys
from multiprocessing.pool import ThreadPool

import numpy as np
import pandas as pd
import xgboost as xgb
from keras.backend import categorical_crossentropy
from keras.callbacks import EarlyStopping
from keras.initializers import RandomNormal, RandomUniform, Zeros
from keras.layers import (Activation, Concatenate, Dense, Dropout, Flatten,
                          Input, InputLayer, Lambda)
from keras.models import Model, Sequential, load_model
from keras.optimizers import Adam, Nadam, RMSprop
from sklearn.impute import SimpleImputer
# from keras.activations import relu
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from tensorflow import nn
from treelib import Node, Tree

from environment import HouseEnv

MAX_EPSILON = 1.0       # 1.0
MIN_EPSILON = 0.01      # 0.01
LAMBDA = 0.00005         # 0.0005
GAMMA = 0.95            # 0.95
BATCH_SIZE = 32         # 32
TAU = 0.08               # 0.08
RANDOM_REWARD_STD = 1.0  # 1.0
LEARNING_RATE = 0.0001   # 0.0003


def relu(x): return nn.leaky_relu(x)


def rmse(x, y): return math.sqrt(mean_squared_error(x, y))


def network_builder(inputs_number=25, activation='linear', dropout=.0, layers=3,
                    ending_units=256, kernel_initializer='random_normal'):
    def generate(units, activation, layers, pred):
        if layers > 1:
            return Dense(units=units, activation=activation, use_bias=True,
                         kernel_initializer=kernel_initializer)(
                Dropout(dropout)(
                    generate(units * 2, activation, layers - 1, pred))
                if dropout else generate(units * 2, activation, layers - 1, pred))
        else:
            return Dense(units=units, activation=activation, use_bias=True,
                         kernel_initializer=kernel_initializer)(pred)

    input = Input(shape=(inputs_number,))
    output = Dense(units=1, activation=activation,  # softmax
                   use_bias=True)(generate(ending_units, relu, layers, input))

    return Model(inputs=[input], outputs=[output])


def diamond_network_builder(inputs_number=18, activation='linear', dropout=.0, layers=3,
                            ending_units=256, kernel_initializer='random_normal'):
    def generate(units, activation, initial_layers, layers, input):
        if layers > 1:
            new_units = units
            if initial_layers % 2:
                if layers > math.ceil(initial_layers / 2):
                    new_units *= 2
                elif layers <= math.ceil(initial_layers / 2):
                    new_units //= 2
            else:
                if layers > math.ceil(initial_layers / 2) + 1:
                    new_units *= 2
                elif layers < math.ceil(initial_layers / 2) + 1:
                    new_units //= 2

            return Dense(units=units, activation=activation, use_bias=True, kernel_initializer=kernel_initializer)(
                Dropout(dropout)(
                    generate(new_units, activation, initial_layers, layers - 1, input))
                if dropout else generate(new_units, activation, initial_layers, layers - 1, input)
            )
        else:
            return Dense(units=units, activation=activation, use_bias=True,
                         kernel_initializer=kernel_initializer)(input)

    input = Input(shape=(inputs_number,))
    output = Dense(units=1, activation=activation,
                   use_bias=True)(generate(ending_units, relu, layers, layers, input))

    return Model(inputs=[input], outputs=[output])


def prepare_data(data, train=True):
    num_imputer = SimpleImputer()
    label_encoder = LabelEncoder()

    y = None
    if train:
        y = data.SalePrice
    num_X = data.select_dtypes(exclude=['object'])
    if train:
        del num_X["SalePrice"]
    num_X_columns = num_X.columns
    num_X = pd.DataFrame(num_imputer.fit_transform(num_X))
    num_X.columns = num_X_columns

    cols_without_missing = [
        col for col in data.columns if not data[col].isnull().any()]
    data = data[cols_without_missing]
    object_cols = [
        col for col in data.columns if data[col].dtype == "object"]
    low_cardinality_cols = [
        col for col in object_cols if data[col].nunique() < 10]
    high_cardinality_cols = list(set(object_cols) - set(low_cardinality_cols))

    label_X = data[high_cardinality_cols]
    for col in high_cardinality_cols:
        label_X[col] = label_encoder.fit_transform(label_X[col])

    X = pd.concat([num_X, label_X], axis=1)

    return X, y

def alt_prepare_data(data, train=True):
    num_imputer = SimpleImputer()
    sub_imputer = SimpleImputer(strategy='most_frequent')
    label_encoder = LabelEncoder()

    y = None
    if train:
        y = data.SalePrice
    num_X = data.select_dtypes(exclude=['object'])
    if train:
        del num_X["SalePrice"]
    num_X_columns = num_X.columns
    num_X = pd.DataFrame(num_imputer.fit_transform(num_X))
    num_X.columns = num_X_columns

    object_cols = [
        col for col in data.columns if data[col].dtype == "object"]
    data_columns = data[object_cols].columns
    data = pd.DataFrame(sub_imputer.fit_transform(data[object_cols]))
    data.columns = data_columns

    low_cardinality_cols = [
        col for col in object_cols if data[col].nunique() < 15]
    high_cardinality_cols = list(set(object_cols) - set(low_cardinality_cols))

    label_X = data[high_cardinality_cols]
    for col in high_cardinality_cols:
        label_X[col] = label_encoder.fit_transform(label_X[col])

    X = pd.concat([num_X, label_X], axis=1)

    return X, y

def prepare_numpy(X, y=None):
    return X.to_numpy(), y.to_numpy() if y is not None else None


def exclude_features(data, features):
    for f in features:
        try:
            del data[f]
        except:
            pass


def norm(X):
    stats = X.describe().transpose()
    return (X - stats['mean']) / stats['std']


class KFold(object):
    def __init__(self, X, y, k=4, epochs=200, loss='mse', metrics=['mae'], learning_rate=0.0005,
                 layers=3, ending_units=256, optimizer=Adam, early_stop=None, seed=None):
        self.X = X
        self.y = y
        self.k = k
        self.epochs = epochs
        self.loss = loss
        self.metrics = metrics
        self.learning_rate = learning_rate
        self.layers = layers
        self.ending_units = ending_units
        self.optimizer = optimizer
        self.early_stop = early_stop
        self.generator = secrets.SystemRandom(seed)
        self._idx = self.generator.randint(0, len(X))
        self.models = []

    @property
    def idx(self):
        return self._idx

    @idx.setter
    def idx(self, value):
        self._idx = value
        if self._idx >= len(self.X):
            self._idx = self._idx - len(self.X)

    @idx.deleter
    def idx(self):
        del self._idx

    def fit(self):
        s = math.ceil(len(X) / self.k)

        for _ in range(self.k):
            delta = max(self.idx - len(X), 0)
            ceil = max(self.idx + s, len(X))
            model = network_builder(
                layers=self.layers, ending_units=self.ending_units)
            model.compile(
                loss=self.loss, optimizer=self.optimizer(learning_rate=self.learning_rate), metrics=self.metrics)
            model.fit(np.concatenate((X[self.idx:ceil], X[0:delta])), np.concatenate((y[self.idx:ceil], y[0:delta])),
                      epochs=self.epochs, callbacks=[self.early_stop] if self.early_stop else [])
            self.models.append(model)

            self.idx += s

    def predict(self, X):
        p = 0.

        for el in self.models:
            p += el.predict(X)

        return p / len(self.models)


class StohasticInitializer(object):
    def __init__(self, X, y, k=4, epochs=200, loss='mse', metrics=['mae'], learning_rate=0.0005,
                 layers=3, ending_units=[128, 256, 512], optimizers=[Adam, Nadam, RMSprop], early_stop=None, seed=None):
        self.X = X
        self.y = y
        self.k = k
        self.epochs = epochs
        self.loss = loss
        self.metrics = metrics
        self.learning_rate = learning_rate
        self.layers = layers
        self.ending_units = ending_units
        self.optimizers = optimizers
        self.early_stop = early_stop
        self.generator = secrets.SystemRandom(seed)
        self.models = []

    def fit(self):
        for _ in range(self.k):
            optimizer = self.optimizers[self.generator.randint(
                0, len(self.optimizers) - 1)](learning_rate=self.learning_rate)
            # layers = self.generator.randint(1, self.layers)
            # ending_units = self.ending_units[self.generator.randint(
            # 0, len(self.ending_units) - 1)]
            model = network_builder(
                layers=self.layers, ending_units=self.ending_units, kernel_initializer="random_uniform")
            model.compile(
                loss=self.loss,
                optimizer=optimizer,
                metrics=self.metrics)
            model.fit(self.X, self.y, epochs=self.epochs,
                      callbacks=[self.early_stop] if self.early_stop else [])
            self.models.append(model)

    def predict(self, X):
        p = 0.

        for el in self.models:
            p += el.predict(X)

        return p / len(self.models)


class GradientBoosting(object):
    def __init__(self, X, y, k=4, epochs=200, loss='mse', metrics=['mae'], learning_rate=0.001,
                 layers=3, ending_units=256, optimizers=[Adam, Nadam, RMSprop], early_stop=None, seed=None):
        self.X = X
        self.y = y
        self.k = k
        self.epochs = epochs
        self.loss = loss
        self.metrics = metrics
        self.learning_rate = learning_rate
        self.layers = layers
        self.ending_units = ending_units
        self.optimizers = optimizers
        self.early_stop = early_stop
        self.generator = secrets.SystemRandom(seed)
        self.models = []

    def fit(self):
        y = self.y
        preds = np.zeros(len(y))

        for i in range(self.k):
            optimizer = self.optimizers[self.generator.randint(
                0, len(self.optimizers) - 1)](learning_rate=self.learning_rate)
            model = network_builder(
                layers=self.layers, ending_units=self.ending_units)
            model.compile(
                loss=self.loss,
                optimizer=optimizer,
                metrics=self.metrics)
            model.fit(self.X, y, epochs=self.epochs * (i + 1),
                      callbacks=[self.early_stop] if self.early_stop else [])
            self.models.append(model)
            preds = preds + model.predict(self.X).reshape(-1)
            y = self.y - preds

    def predict(self, X):
        p = 0.

        for el in self.models:
            p += el.predict(X)

        return p


class GradientBoostingTree(object):
    def __init__(self, X, y, k=4, epochs=200, loss='mse', metrics=['mae'], learning_rate=0.001,
                 layers=3, ending_units=256, optimizers=[Adam, Nadam, RMSprop], early_stop=None, seed=None):
        self.X = X
        self.y = y
        self.k = k
        self.epochs = epochs
        self.loss = loss
        self.metrics = metrics
        self.learning_rate = learning_rate
        self.layers = layers
        self.ending_units = ending_units
        self.optimizers = optimizers
        self.early_stop = early_stop
        self.generator = secrets.SystemRandom(seed)
        self.models = []
        self.tree = Tree()

    def fit(self):
        y = self.y
        preds = np.zeros(len(y))

        def fit_leaf(preds, y, i):
            optimizer = self.optimizers[self.generator.randint(
                0, len(self.optimizers) - 1)](learning_rate=self.learning_rate)
            model = network_builder(
                layers=self.layers, ending_units=self.ending_units)
            model.compile(
                loss=self.loss,
                optimizer=optimizer,
                metrics=self.metrics)
            model.fit(self.X, y, epochs=self.epochs * (i + 1),
                      callbacks=[self.early_stop] if self.early_stop else [])
            new_preds = preds + model.predict(self.X).reshape(-1)
            new_y = self.y - new_preds

            return new_preds, new_y, model

        def fit_tree(k, preds, y, i, parent=None):
            if k > 0:
                l_preds, l_y, l_model = fit_leaf(preds, y, i)
                self.tree.add_node(Node(identifier=parent + 'l' + str(k),
                                        data=l_model), parent=parent)
                fit_tree(k - 1, l_preds, l_y, i + 1, parent + 'l' + str(k))

                r_preds, r_y, r_model = fit_leaf(preds, y, i)
                self.tree.add_node(Node(identifier=parent + 'r' + str(k),
                                        data=r_model), parent=parent)
                fit_tree(k - 1, r_preds, r_y, i + 1, parent + 'r' + str(k))

        i = 0
        preds, y, root_model = fit_leaf(preds, y, i)
        self.tree.add_node(Node(identifier='root', data=root_model))
        fit_tree(self.k, preds, y, i + 1, parent='root')

    def predict(self, X):
        preds = []
        leafs = self.tree.leaves('root')
        for path in self.tree.paths_to_leaves():
            p = .0
            for id in path:
                p += self.tree.get_node(id).data.predict(X)
            preds.append(p)

        return sum(preds) / len(leafs)


early_stop = EarlyStopping(monitor='val_loss', patience=10)


iowa_file_path = 'train.csv'
test_data_path = 'test.csv'
home_data = pd.read_csv(iowa_file_path)
test_data = pd.read_csv(test_data_path)

X, y = alt_prepare_data(home_data)
# print([col for col in X])
# sys.exit(0)
test_X, _ = alt_prepare_data(test_data, False)
common_columns = set(X.columns) & set(test_X.columns)
X = X[common_columns]
test_X = test_X[common_columns]
ids = test_X.Id.astype(int)
features_to_exclude = [
    "Id", "YrSold", "MoSold", "BsmtFinSF2", "BsmtUnfSF", "MiscVal",
    "BedroomAbvGr", "MasVnrArea", "ScreenPorch", "OpenPorchSF", "MSSubClass", "EnclosedPorch",
    "3SsnPorch", "Exterior1st", "Exterior2nd"
]
#     "Id", "MasVnrArea", "WoodDeckSF", "BedroomAbvGr", "MoSold", "LotFrontage",
#     "LowQualFinSF", "YrSold", "ScreenPorch", "OpenPorchSF", "MSSubClass", "EnclosedPorch",
#     "3SsnPorch", "BsmtFinSF2", "Exterior1st", "Exterior2nd", "TotRmsAbvGrd", "GarageYrBlt",
#     "HalfBath", "BsmtFullBath", "BsmtHalfBath", "BsmtUnfSF"
# ]
exclude_features(X, features_to_exclude)
exclude_features(test_X, features_to_exclude)
# print([col for col in X])
# sys.exit(0)

X = norm(X)
test_X = norm(test_X)
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=0)
# model = network_builder()
# model.compile(
#     loss='mse', optimizer=Adam(learning_rate=LEARNING_RATE), metrics=['mae'])
nX, ny = prepare_numpy(X, y)
ntrain_X, ntrain_y = prepare_numpy(train_X, train_y)
nval_X, nval_y = prepare_numpy(val_X, val_y)
ntest_X, _ = prepare_numpy(test_X)

# kfold = KFold(nX, ny, k=8, epochs=200, layers=3,
#               ending_units=128, early_stop=early_stop)
# kfold.fit()
# preds1 = kfold.predict(ntest_X).reshape(-1)
# si = StohasticInitializer(nX, ny, k=8, epochs=200,
#                           layers=3, ending_units=256, early_stop=early_stop)
# si.fit()
# preds2 = si.predict(ntest_X).reshape(-1)
# gb = GradientBoostingTree(nX, ny, k=4,
#                           epochs=20, layers=3, ending_units=128)
# gb.fit()
# preds3 = gb.predict(ntest_X).reshape(-1)


gbs = []
preds = []
for _ in range(6):
    gbs.append(GradientBoosting(nX, ny, k=5,
                                epochs=20, layers=3, ending_units=128))
    gbs[-1].fit()
    preds.append(gbs[-1].predict(ntest_X).reshape(-1))

# model = xgb.XGBRegressor(n_estimators=750, learning_rate=0.02, silent=1,
#                          objective='reg:squarederror', nthread=-1, subsample=0.7,
#                          colsample_bytree=0.7, max_depth=6, seed=0, n_jobs=-1)

# model.fit(train_X, train_y,
#           early_stopping_rounds=5,
#           eval_set=[(val_X, val_y)],
#           verbose=False)
# preds4 = model.predict(test_X)

# preds = (preds1 + preds2 + preds3 + preds4) / 4
# print("Validation MAE: {:,.0f}".format(
#     mean_absolute_error(nval_y, sum(preds) / len(preds))))

# model.fit(nX, ny, epochs=500, callbacks=[early_stop])
# print(model.evaluate(nval_X, nval_y))
# preds = model.predict(ntest_X).reshape(-1)

output = pd.DataFrame({'Id': ids,
                       'SalePrice': sum(preds) / len(preds)})
output.to_csv('submission.csv', index=False)
