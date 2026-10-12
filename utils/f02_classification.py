from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, mutual_info_score


def mutual_info_scores(df, features, y, decimals=2):
    """
    Calculate the mutual information score between the target and each feature.

    Args:
        df (pd.DataFrame): The dataset holding the features.
        features (list): The columns to score.
        y (np.ndarray): The target vector.
        decimals (int, optional): Decimals to round the scores to. Defaults to 2.

    Returns:
        pd.Series: The scores, rounded and sorted in descending order.
    """
    scores = df[features].apply(lambda series: mutual_info_score(series, y))
    return scores.round(decimals).sort_values(ascending=False)


def train_logistic_regression(df_train, y_train, features, C=1.0):
    """
    Train a logistic regression model, one-hot encoding the categorical features.

    Args:
        df_train (pd.DataFrame): The training dataset.
        y_train (np.ndarray): The training target vector.
        features (list): The columns to use as features.
        C (float, optional): The inverse of the regularization strength. Defaults to 1.0.

    Returns:
        tuple: A tuple containing the fitted DictVectorizer and the fitted model.
    """
    dv = DictVectorizer(sparse=False)
    X_train = dv.fit_transform(df_train[features].to_dict(orient='records'))

    model = LogisticRegression(solver='liblinear', C=C, max_iter=1000, random_state=42)
    model.fit(X_train, y_train)

    return dv, model


def predict(df, features, dv, model):
    """
    Predict the target for a dataset using a fitted vectorizer and model.

    Args:
        df (pd.DataFrame): The dataset to predict on.
        features (list): The columns to use as features.
        dv (DictVectorizer): The fitted vectorizer.
        model (LogisticRegression): The fitted model.

    Returns:
        np.ndarray: The predicted classes.
    """
    X = dv.transform(df[features].to_dict(orient='records'))
    return model.predict(X)


def accuracy_for_features(df_train, y_train, df_val, y_val, features, C=1.0):
    """
    Train a logistic regression on a set of features and score it on the validation set.

    Args:
        df_train (pd.DataFrame): The training dataset.
        y_train (np.ndarray): The training target vector.
        df_val (pd.DataFrame): The validation dataset.
        y_val (np.ndarray): The validation target vector.
        features (list): The columns to use as features.
        C (float, optional): The inverse of the regularization strength. Defaults to 1.0.

    Returns:
        float: The accuracy on the validation dataset.
    """
    dv, model = train_logistic_regression(df_train, y_train, features, C)
    return accuracy_score(y_val, predict(df_val, features, dv, model))
