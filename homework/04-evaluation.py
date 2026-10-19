import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from utils.f00_logger import logger
from utils.f02_classification import predict_proba, train_logistic_regression
from utils.f03_evaluation import auc_for_features, cross_validate_auc, threshold_scores

path_data = 'homework/data/course_lead_scoring_2026.csv'

df = pd.read_csv(path_data)
logger.info(f'Dataset read! The shape is {df.shape[0]} rows and {df.shape[1]} columns')

# Preparation
target = 'converted'
features = [col for col in df.columns if col != target]

categorical = df[features].select_dtypes(include='str').columns.tolist()
numerical = df[features].select_dtypes(include='number').columns.tolist()
logger.info(f'Categorical features: {categorical}')
logger.info(f'Numerical features: {numerical}')

df[categorical] = df[categorical].fillna('NA')
df[numerical] = df[numerical].fillna(0.0)
logger.info(f'Missing values after filling them: {df.isnull().sum().sum()}')

# Split the data
df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)
df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=1)

df_full_train = df_full_train.reset_index(drop=True)
df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

y_train = df_train[target].values
y_val = df_val[target].values
y_test = df_test[target].values

logger.info(f'Train size: {len(df_train)}, Validation size: {len(df_val)}, Test size: {len(df_test)}')

# Question 1
auc_scores = auc_for_features(df_train, numerical, y_train)
logger.info(f'AUC of each numerical feature: \n{auc_scores.round(4)}')
logger.info(f'Highest AUC: {auc_scores.index[0]}')

# Question 2
dv, model = train_logistic_regression(df_train, y_train, features)
y_pred = predict_proba(df_val, features, dv, model)

auc = roc_auc_score(y_val, y_pred)
logger.info(f'AUC on the validation dataset: {round(auc, 3)}')

# Question 3
scores = threshold_scores(y_val, y_pred)

intersection = scores[(scores.precision > 0) | (scores.recall > 0)].copy()
intersection['difference'] = (intersection.precision - intersection.recall).abs()
intersection = intersection.loc[intersection.difference.idxmin()]
logger.info(
    f'Precision and recall intersect at threshold {intersection.threshold}: '
    f'precision = {intersection.precision}, recall = {intersection.recall}'
)

# Question 4
best_f1 = scores.loc[scores.f1.idxmax()]
logger.info(f'F1 is maximal at threshold {best_f1.threshold} with a value of {best_f1.f1}')

# Question 5
fold_scores = cross_validate_auc(df_full_train, target, features)
logger.info(f'AUC on each fold: {[round(score, 4) for score in fold_scores]}')
logger.info(f'Mean AUC: {round(float(np.mean(fold_scores)), 3)}')
logger.info(f'Standard deviation of the scores: {round(float(np.std(fold_scores)), 3)}')

# Question 6
results = {}
for C in [0.000001, 0.001, 1]:
    fold_scores = cross_validate_auc(df_full_train, target, features, C)
    results[C] = (round(float(np.mean(fold_scores)), 3), round(float(np.std(fold_scores)), 3))
    logger.info(f'C = {C}: mean AUC = {results[C][0]}, std = {results[C][1]}')

best_c = min(results, key=lambda c: (-results[c][0], results[c][1], c))
logger.info(f'Best C is {best_c} with a mean AUC of {results[best_c][0]}')
