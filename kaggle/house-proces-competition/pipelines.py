import itertools
import os

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


iowa_file_path = 'train.csv'
test_data_path = 'test.csv'
home_data = pd.read_csv(iowa_file_path)
test_data = pd.read_csv(test_data_path)

y = home_data.SalePrice
X = home_data.copy()
del X["SalePrice"]

numerical_transformer = SimpleImputer(strategy='constant')
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

numerical_cols = [
    col for col in X.columns if X[col].dtype != "object"]
categorical_cols = [
    col for col in X.columns if X[col].dtype == "object"]

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ]
)

model = RandomForestRegressor(n_estimators=100, random_state=0)

my_pipeline = Pipeline(
    steps=[
        ('preprocessor', preprocessor),
        ('model', model)
    ]
)

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=0)

my_pipeline.fit(train_X, train_y)

preds = my_pipeline.predict(val_X)

score = mean_absolute_error(val_y, preds)
print('MAE:', score)
