import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold

from utils.f02_classification import predict_proba, train_logistic_regression


def auc_for_features(df, features, y):
    """
    Use each feature as a score on its own and compute its ROC AUC against the target.

    A feature with an AUC below 0.5 is negatively correlated with the target, so it is
    negated to flip the direction of the correlation.

    Args:
        df (pd.DataFrame): The dataset holding the features.
        features (list): The numerical columns to score.
        y (np.ndarray): The target vector.

    Returns:
        pd.Series: The AUC of each feature, sorted in descending order.
    """
    scores = {}
    for feature in features:
        auc = roc_auc_score(y, df[feature])
        scores[feature] = roc_auc_score(y, -df[feature]) if auc < 0.5 else auc

    return pd.Series(scores).sort_values(ascending=False)


def threshold_scores(y, y_pred, step=0.01):
    """
    Compute the confusion matrix and its metrics for every threshold in [0, 1].

    Args:
        y (np.ndarray): The target vector.
        y_pred (np.ndarray): The predicted probability of the positive class.
        step (float, optional): The increment between thresholds. Defaults to 0.01.

    Returns:
        pd.DataFrame: One row per threshold with tp, fp, fn, tn, precision, recall and f1.
    """
    rows = []
    for threshold in np.arange(0.0, 1.0 + step, step):
        predict_positive = y_pred >= threshold

        tp = ((predict_positive) & (y == 1)).sum()
        fp = ((predict_positive) & (y == 0)).sum()
        fn = ((~predict_positive) & (y == 1)).sum()
        tn = ((~predict_positive) & (y == 0)).sum()

        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

        rows.append((round(threshold, 2), tp, fp, fn, tn, precision, recall, f1))

    columns = ['threshold', 'tp', 'fp', 'fn', 'tn', 'precision', 'recall', 'f1']
    return pd.DataFrame(rows, columns=columns)


def cross_validate_auc(df_full_train, target, features, C=1.0, n_splits=5, random_state=1):
    """
    Evaluate the model with K-fold cross-validation using ROC AUC.

    Args:
        df_full_train (pd.DataFrame): The dataset to split into folds.
        target (str): The name of the target column.
        features (list): The columns to use as features.
        C (float, optional): The inverse of the regularization strength. Defaults to 1.0.
        n_splits (int, optional): The number of folds. Defaults to 5.
        random_state (int, optional): Seed for the fold shuffling. Defaults to 1.

    Returns:
        list: The AUC obtained on each fold.
    """
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    scores = []
    for train_idx, val_idx in kfold.split(df_full_train):
        df_train = df_full_train.iloc[train_idx]
        df_val = df_full_train.iloc[val_idx]

        dv, model = train_logistic_regression(df_train, df_train[target].values, features, C)
        y_pred = predict_proba(df_val, features, dv, model)

        scores.append(roc_auc_score(df_val[target].values, y_pred))

    return scores
