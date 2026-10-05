import numpy as np


def split_dataset(df, seed, val_size=0.2, test_size=0.2):
    """
    Split the dataset into train, validation, and test sets.

    Args:
        df (pd.DataFrame): The dataset to split.
        seed (int): The seed for the random number generator.
        val_size (float, optional): Share of rows for validation. Defaults to 0.2.
        test_size (float, optional): Share of rows for test. Defaults to 0.2.

    Returns:
        tuple: A tuple containing the train, validation, and test sets.
    """
    n = len(df)
    n_val = int(n * val_size)
    n_test = int(n * test_size)
    n_train = n - n_val - n_test

    np.random.seed(seed)
    idx = np.arange(n)
    np.random.shuffle(idx)

    df_train = df.iloc[idx[:n_train]].reset_index(drop=True)
    df_val = df.iloc[idx[n_train:n_train + n_val]].reset_index(drop=True)
    df_test = df.iloc[idx[n_train + n_val:]].reset_index(drop=True)
    return df_train, df_val, df_test


def prepare_X(df, features, fill_value=0):
    """
    Prepare the feature matrix for linear regression.

    Args:
        df (pd.DataFrame): The dataset to prepare.
        features (list): The columns to use as features.
        fill_value (int, optional): The value to fill missing values with. Defaults to 0.

    Returns:
        np.ndarray: The feature matrix.
    """
    return df[features].fillna(fill_value).values


def train_linear_regression(X, y, r=0.0):
    """
    Train a linear regression model with regularization.

    Args:
        X (np.ndarray): The feature matrix.
        y (np.ndarray): The target vector.
        r (float, optional): The regularization parameter. Defaults to 0.0.

    Returns:
        tuple: A tuple containing the intercept and coefficients of the linear regression model.
    """
    X = np.column_stack([np.ones(X.shape[0]), X])
    XTX = X.T.dot(X) + r * np.eye(X.shape[1])
    w_full = np.linalg.inv(XTX).dot(X.T).dot(y)
    return w_full[0], w_full[1:]


def rmse(y, y_pred):
    """
    Calculate the root mean squared error (RMSE) between the true values and the predicted values.

    Args:
        y (np.ndarray): The true values.
        y_pred (np.ndarray): The predicted values.

    Returns:
        float: The root mean squared error.
    """
    return np.sqrt(((y - y_pred) ** 2).mean())
