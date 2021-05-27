import numpy as np
import matplotlib.pyplot as plt


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


def plot_data(x: np.array, y: np.array):
    """
    plots the input data with 2 markers for the different labels

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


def sigmoid(z: np.array) -> np.array:
    """
    Applies a sigmoid function over the input

    Args:
        z: input array. in the Logistic Regression context should be  theta.T @ x

    Returns:
        h: sigmoid function applied over z
    """
    sig = 1 / (1 + np.exp(-z))
    return sig


def train_test_split(x: np.array, y: np.array, ratio: float = 0.8) -> tuple:
    """
    split the data into train and test batches, according to a given ratio

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
    data_len = x.shape[0]
    train_len = int(np.round(ratio * data_len))
    indices = np.random.permutation(data_len)
    train_idx, test_idx = indices[:train_len], indices[train_len:]
    x_train, x_test = x[train_idx, :], x[test_idx, :]
    y_train, y_test = y[train_idx, :], y[test_idx, :]
    return x_train, y_train, x_test, y_test


def plot_cost(cost_list: list):
    """
    plots the cost function over iterations

    Args:
        cost_list: list of all the cost values
    """
    plt.plot(cost_list)
    plt.xlabel("iterations")
    plt.ylabel("cost function")
    plt.show()


def h(theta: np.array, x: np.array) -> np.array:
    """
    computes the hypothesis function

    Args:
        x: input data
        theta: logistic regression's parameters
    Returns:
        h: sigmoid function applied over z=x@theta
    """
    hypothesis = sigmoid(x @ theta.T)
    return hypothesis


def cost_function(theta: np.array, x: np.array, y: np.array, llambda: float = 0.) -> float:
    """
    calculates the cost function with given regularization argument

    Args:
        theta: theta vector of the model
        x: input data
        y: target
        llambda: lambda of the regularization argument (default 0.)

    Returns:
        cost: cost function's value
    """
    m = len(y)
    hypo = h(theta, x)
    log_h = np.log(hypo)
    cost_term = (-y.T @ log_h - (1 - y).T @ (np.log(1 - hypo)))  # -y*log(h)-(1-y)log(1-h)
    reg_term = (llambda / 2) * (theta[1:].T @ theta[1:])  # no need for theta[0]
    cost = ((cost_term + reg_term) / m).item()
    return cost
