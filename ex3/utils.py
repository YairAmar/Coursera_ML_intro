import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from scipy.io import loadmat


def load_data(path: str, add_bias: bool = True) -> tuple:
    """
    Loading data and formatting for the latter linear regression

    Args:
        path: directory path of the csv file containing the data
        add_bias: if True x will be returned with 1 in the first column
    Returns:
        x: data features
        y: data classes
    """
    data_path = path
    data = loadmat(data_path)
    x = np.array(data["X"])

    if add_bias:
        poly = PolynomialFeatures(1)
        x = poly.fit_transform(x)

    y = np.array(data["y"])
    y[y == 10] = 0
    return x, y


def sigmoid(z: np.ndarray) -> np.ndarray:
    """
    Applies a sigmoid function over the input

    Args:
        z: input array. in the Logistic Regression context should be  theta.T @ x

    Returns:
        h: sigmoid function applied over z
    """
    sig = 1 / (1 + np.exp(-z))
    return sig


def train_test_split(x: np.ndarray, y: np.ndarray, ratio: float = 0.8) -> tuple:
    """
    split the data into train and test batches, according to a given ratio

    Args:
        x: input data
        y: data classes
        ratio: ration between train and test (default 0.8 - 80% train)

    Returns:
        x_train: train data
        y_train: train classes
        x_test: test data
        y_test: test classes
    """
    data_len = x.shape[0]
    train_len = int(np.round(ratio * data_len))
    indices = np.random.permutation(data_len)
    train_idx, test_idx = indices[:train_len], indices[train_len:]
    x_train, x_test = x[train_idx, :], x[test_idx, :]
    y_train, y_test = y[train_idx, :], y[test_idx, :]
    return x_train, y_train, x_test, y_test
