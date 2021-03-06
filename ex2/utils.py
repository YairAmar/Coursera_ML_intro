import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures


def load_data(path: str, add_bias: bool = True) -> tuple:
    """
    Loading data and formatting for the latter linear regression

    Args:
        path: directory path of the csv file containing the data
        add_bias: if True x will be returned with 1 in the first column
    Returns:
        x: data features
        y: data labels
    """
    data1 = np.loadtxt(path, delimiter=',', unpack=True)
    x = np.array(data1[:-1]).T
    y = np.array(data1[-1:]).T
    if add_bias:
        x = np.insert(x, 0, 1, axis=1)
    return x, y


def plot_data(x: np.ndarray, y: np.ndarray):
    """
    Plots the input data with 2 markers for the different labels.

    Args:
        x: input data
        y: data labels
    """
    x1 = x[:, 0]
    x2 = x[:, 1]
    plt.scatter(x1[y[:, 0] == 1], x2[y[:, 0] == 1], marker='+')
    plt.scatter(x1[y[:, 0] == 0], x2[y[:, 0] == 0], marker='3')
    plt.xlabel("Exam 1 score")
    plt.ylabel("Exam 2 score")
    plt.legend(["Admitted", "Not admitted"])


def sigmoid(z: np.ndarray) -> np.ndarray:
    """
    Applies a sigmoid function over the input.

    Args:
        z: in the Logistic Regression context should be  theta.T @ x

    Returns:
        h: sigmoid function applied over z
    """
    sig = 1 / (1 + np.exp(-z))
    return sig


def train_test_split(x: np.ndarray, y: np.ndarray, ratio: float = 0.8) -> tuple:
    """
    Splits the data into 2 groups - train and test, in a proportion given by ratio.

    Args:
        x: input data
        y: data labels
        ratio: ration between train and test (default 0.8 - 80% train)

    Returns:
        x_train: train data
        y_train: train labels
        x_test: test data
        y_test: test labels
    """
    m = x.shape[0]
    train_len = int(np.round(ratio * m))
    indices = np.random.permutation(m)
    train_idx, test_idx = indices[:train_len], indices[train_len:]
    x_train, x_test = x[train_idx, :], x[test_idx, :]
    y_train, y_test = y[train_idx, :], y[test_idx, :]
    return x_train, y_train, x_test, y_test


def plot_cost(cost_list: list):
    """
    Plots all values of the cost-function acquired through the training of the model.

    Args:
        cost_list: list of all cost-function values acquired.
    """
    plt.plot(cost_list)
    plt.xlabel("iterations")
    plt.ylabel("cost function")
    plt.show()


def compute_hypothesis(theta: np.ndarray, x: np.ndarray) -> np.ndarray:
    """
    Computes the hypothesis function.

    Args:
        x: input data
        theta: logistic regression's parameters
    Returns:
        hypothesis: sigmoid function applied over z=x@theta
    """
    hypothesis = sigmoid(x @ theta.T)
    return hypothesis


def compute_cost_function(theta: np.ndarray, x: np.ndarray, y: np.ndarray, llambda: float = 0.) -> float:
    """
    Calculates the cost function with given regularization argument.

    Args:
        theta: theta vector of the model
        x: input data
        y: target
        llambda: lambda of the regularization argument (default 0.)

    Returns:
        cost: cost function's value
    """
    m = len(y)
    hypo = compute_hypothesis(theta, x)
    log_h = np.log(hypo)
    cost_term = (-y.T @ log_h - (1 - y).T @ (np.log(1 - hypo)))  # -y*log(h)-(1-y)log(1-h)
    reg_term = (llambda / 2) * (theta[1:].T @ theta[1:])  # no need for theta[0]
    cost = ((cost_term + reg_term) / m).item()
    return cost


def pre_process(file_path: str, poly_deg: int = 1) -> tuple:
    """
    Plots the input data and creates polynomial features for the data.
    Args:
        file_path: path to the file from which the data should be imported
        poly_deg: degree of polynomial features required (default 1)

    Returns:
        x_train: train data
        y_train: train labels
        x_test: test data
        y_test: test labels
    """
    x, y = load_data(file_path, add_bias=False)
    plot_data(x, y)
    plt.show()
    poly = PolynomialFeatures(poly_deg)
    x = poly.fit_transform(x)
    x_train, y_train, x_test, y_test = train_test_split(x, y)
    #x_train[:, 1:] = (x_train[:, 1:] - np.mean(x_train[:, 1:], axis=0)) / np.std(x_train[:, 1:], axis=0)
    #x_test[:, 1:] = (x_test[:, 1:] - np.mean(x_test[:, 1:], axis=0)) / np.std(x_test[:, 1:], axis=0)
    return x_train, y_train, x_test, y_test
