import itertools
import os
from math import sqrt

import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


def rmse(x, y): return sqrt(mean_squared_error(x, y))


def normilize(df):
    return (df - df.min()) / (df.max() - df.min())


def exclude_features(data, features):
    for f in features:
        try:
            del data[f]
        except:
            pass


def prepare_data(data, train=True):
    num_imputer = SimpleImputer()
    # cat_imputer = SimpleImputer(strategy='most_frequent')
    label_encoder = LabelEncoder()
    # oh_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)

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
    # data = pd.DataFrame(cat_imputer.fit_transform(data[object_cols]))
    # data.columns = object_cols

    # oh_X = pd.DataFrame(oh_encoder.fit_transform(data[low_cardinality_cols]))
    # oh_X.index = data.index

    label_X = data[high_cardinality_cols]
    for col in high_cardinality_cols:
        label_X[col] = label_encoder.fit_transform(label_X[col])

    X = pd.concat([num_X, label_X], axis=1)

    return X, y


def features_estimator(features, train_X, train_y, val_X, val_y):
    min_error = 10. ** 5
    best_features_combination = []
    rf_model = RandomForestRegressor(
        random_state=0, n_estimators=600, max_leaf_nodes=750, min_samples_split=3, n_jobs=-1)
    for i in range(5, len(features)):
        for c in itertools.combinations(features, i):
            rf_model.fit(train_X, train_y)
            preds = rf_model.predict(val_X)
            error = mean_absolute_error(val_y, preds)
            if error < min_error:
                best_features_combination = c
                min_error = error

    return best_features_combination, min_error


iowa_file_path = 'train.csv'
test_data_path = 'test.csv'
home_data = pd.read_csv(iowa_file_path)
test_data = pd.read_csv(test_data_path)

X, y = prepare_data(home_data)
test_X, _ = prepare_data(test_data, False)
common_columns = set(X.columns) & set(test_X.columns)
X = X[common_columns]
test_X = test_X[common_columns]
ids = test_X.Id.astype(int)
features_to_exclude = [
    "Id", "MasVnrArea", "WoodDeckSF", "BedroomAbvGr", "MoSold", "LotFrontage",
    "LowQualFinSF", "YrSold", "ScreenPorch", "OpenPorchSF", "MSSubClass", "EnclosedPorch",
    "3SsnPorch", "BsmtFinSF2", "Exterior1st", "Exterior2nd", "TotRmsAbvGrd", "GarageYrBlt",
    "HalfBath", "BsmtFullBath", "BsmtHalfBath", "BsmtUnfSF"
]
exclude_features(X, features_to_exclude)
exclude_features(test_X, features_to_exclude)

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=0)

model = RandomForestRegressor(
    random_state=0, n_estimators=600, max_leaf_nodes=750, min_samples_split=2, n_jobs=-1)

model = xgb.XGBRegressor(n_estimators=750, learning_rate=0.02, silent=1,
                         objective='reg:squarederror', nthread=-1, subsample=0.7,
                         colsample_bytree=0.7, max_depth=6, seed=0, n_jobs=-1)
# xg_train = xgb.DMatrix(X, y)
# params = {'eta': 0.02, 'max_depth': 6, 'subsample': 0.7, 'colsample_bytree': 0.7, 'objective': 'reg:squarederror',
#           'seed': 0, 'silent': 1, 'eval_metric': 'rmse', 'nthread': -1}
# scores = xgb.cv(params, xg_train, num_boost_round=2000,
#                 nfold=5, early_stopping_rounds=5)
# print("Cross validation scores:\n{}".format(scores.tail()))

model.fit(train_X, train_y,
          early_stopping_rounds=5,
          eval_set=[(val_X, val_y)],
          verbose=False)
preds = model.predict(val_X)

# scores = -1 * cross_val_score(model, X, y,
#                               cv=5,
#                               n_jobs=-1,
#                               scoring='neg_root_mean_squared_error')
# print("Cross validation MAE: {}".format(scores))

# model.fit(train_X, train_y)
# preds = model.predict(val_X)

print("Validation RMSE: {:,.0f}".format(
    mean_absolute_error(val_y, preds)))

# output = pd.DataFrame({'Id': ids,
#                        'SalePrice': preds})
# output.to_csv('submission.csv', index=False)
