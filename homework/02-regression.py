import pandas as pd
import numpy as np
from utils.f00_logger import logger
from utils.f01_regression import split_dataset, prepare_X, train_linear_regression, rmse

file_path = 'homework/data/car_fuel_efficiency_2026.csv'
df = pd.read_csv(file_path)
logger.info(f"Dataset read! The shape is {df.shape[0]} rows and {df.shape[1]} columns")

cols = ['engine_displacement', 'horsepower', 'vehicle_weight', 'model_year', 'fuel_efficiency_mpg']
df = df[cols]
logger.info(f"Dataset after filtering columns: {df.shape[0]} rows and {df.shape[1]} columns")

df.head()

# Question 1
logger.info(f"Null values in the dataset: \n{df.isnull().sum()}")

# Question 2
logger.info(f"Horserpower statistics: \n{df.horsepower.describe()}")

# Prepare and split the dataset
target = 'fuel_efficiency_mpg'
features = [col for col in cols if col != target]

df_train, df_val, df_test = split_dataset(df, 42)

logger.info(f"Train shape: {df_train.shape}")
logger.info(f"Validation shape: {df_val.shape}")
logger.info(f"Test shape: {df_test.shape}")

y_train = df_train[target].values
y_val = df_val[target].values

# Question 3
logger.info("Question 3")
horsepower_mean = df_train.horsepower.mean()
logger.info(f"Horsepower mean (train only): {horsepower_mean}")

scores_fill = {}
for label, fill_value in [('0', 0), ('mean', horsepower_mean)]:
    w0, w = train_linear_regression(prepare_X(df_train, features, fill_value), y_train)
    y_pred = w0 + prepare_X(df_val, features, fill_value).dot(w)
    scores_fill[label] = round(rmse(y_val, y_pred), 3)
    logger.info(f"Filling horsepower with {label}: RMSE = {scores_fill[label]}")

if scores_fill['0'] < scores_fill['mean']:
    logger.info("Best option: with 0")
elif scores_fill['mean'] < scores_fill['0']:
    logger.info("Best option: with mean")
else:
    logger.info("Best option: both are equally good")

# Question 4
logger.info("Question 4")
X_train = prepare_X(df_train, features, 0)
X_val = prepare_X(df_val, features, 0)

scores_r = {}
for r in [0, 0.01, 0.1, 1, 5, 10, 100]:
    w0, w = train_linear_regression(X_train, y_train, r)
    y_pred = w0 + X_val.dot(w)
    scores_r[r] = round(rmse(y_val, y_pred), 4)
    logger.info(f"r = {r}: RMSE = {scores_r[r]}")

best_rmse = min(scores_r.values())
best_r = min(r for r, score in scores_r.items() if score == best_rmse)
logger.info(f"Best r is {best_r} with RMSE {best_rmse}")

# Question 5
logger.info("Question 5")
scores_seed = []
for seed in range(10):
    df_train_s, df_val_s, df_test_s = split_dataset(df, seed)
    w0, w = train_linear_regression(prepare_X(df_train_s, features, 0), df_train_s[target].values)
    y_pred = w0 + prepare_X(df_val_s, features, 0).dot(w)
    score = rmse(df_val_s[target].values, y_pred)
    scores_seed.append(score)
    logger.info(f"Seed {seed}: RMSE = {score}")

std_scores = round(float(np.std(scores_seed)), 3)
logger.info(f"Standard deviation of the scores: {std_scores}")

# Question 6
logger.info("Question 6")
df_train_9, df_val_9, df_test_9 = split_dataset(df, 9)
df_full_train = pd.concat([df_train_9, df_val_9]).reset_index(drop=True)

w0, w = train_linear_regression(prepare_X(df_full_train, features, 0), df_full_train[target].values, 0.001)
y_pred = w0 + prepare_X(df_test_9, features, 0).dot(w)
test_rmse = rmse(df_test_9[target].values, y_pred)
logger.info(f"RMSE on the test dataset: {round(test_rmse, 3)}")

