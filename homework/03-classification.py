import pandas as pd
from sklearn.model_selection import train_test_split

from utils.f00_logger import logger
from utils.f02_classification import accuracy_for_features, mutual_info_scores

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

mv = df.isnull().sum()
logger.info(f'Missing values in the features: \n{mv}')

df[categorical] = df[categorical].fillna('NA')
df[numerical] = df[numerical].fillna(0.0)
logger.info(f'Missing values after filling them: {df.isnull().sum().sum()}')

# Question 1
mode_industry = df.industry.mode()[0]
logger.info(f'Mode of industry: {mode_industry}')

# Question 2
correlation_matrix = df[numerical].corr()
logger.info(f'Correlation matrix: \n{correlation_matrix.round(4)}')

pairs = [
    ('interaction_count', 'lead_score'),
    ('number_of_courses_viewed', 'lead_score'),
    ('number_of_courses_viewed', 'interaction_count'),
    ('annual_income', 'interaction_count'),
]

correlations = {pair: correlation_matrix.loc[pair] for pair in pairs}
for pair, correlation in correlations.items():
    logger.info(f'Correlation between {pair[0]} and {pair[1]}: {correlation}')

biggest_pair = max(correlations, key=correlations.get)
logger.info(f'Biggest correlation: {biggest_pair[0]} and {biggest_pair[1]}')

# Split the data
df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=42)
df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=42)

df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

y_train = df_train[target].values
y_val = df_val[target].values
y_test = df_test[target].values

del df_train[target]
del df_val[target]
del df_test[target]

logger.info(f'Train size: {len(df_train)}, Validation size: {len(df_val)}, Test size: {len(df_test)}')


# Question 3
mi_scores = mutual_info_scores(df_train, categorical, y_train)
logger.info(f'Mutual information scores: \n{mi_scores}')
logger.info(f'Biggest mutual information score: {mi_scores.index[0]}')

# Question 4
accuracy = accuracy_for_features(df_train, y_train, df_val, y_val, features)
logger.info(f'Accuracy on the validation dataset: {round(accuracy, 2)}')

# Question 5
differences = {}
for feature in features:
    features_without = [f for f in features if f != feature]
    accuracy_without = accuracy_for_features(df_train, y_train, df_val, y_val, features_without)
    differences[feature] = accuracy - accuracy_without
    logger.info(f'Without {feature}: accuracy = {accuracy_without}, difference = {differences[feature]}')

options = ['lead_source', 'number_of_courses_viewed', 'interaction_count']
least_useful = min(options, key=lambda feature: differences[feature])
logger.info(f'Least useful feature: {least_useful} with a difference of {differences[least_useful]}')

# Question 6
scores_c = {}
for C in [0.000001, 0.00001, 0.0001, 0.001]:
    scores_c[C] = round(accuracy_for_features(df_train, y_train, df_val, y_val, features, C), 3)
    logger.info(f'C = {C}: accuracy = {scores_c[C]}')

best_accuracy = max(scores_c.values())
best_c = min(c for c, score in scores_c.items() if score == best_accuracy)
logger.info(f'Best C is {best_c} with an accuracy of {best_accuracy}')
