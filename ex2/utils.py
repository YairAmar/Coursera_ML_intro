import numpy as np
import matplotlib.pyplot as plt


def load_data(path: str) -> tuple:
    """
    Loading data and formatting for the latter linear regression

    Args:
        path: directory path of the csv file containing the data

    Returns:
        x: data features
        y: data labels
    """
    print(path)
    data1 = np.loadtxt(path, delimiter=',', unpack=True)
    x = np.array(data1[:-1]).T
    y = np.array(data1[-1:]).T
    x = np.insert(x, 0, 1, axis=1)
    return x, y


def plot_data(x: np.array, y: np.array):
    """
    plots the input data with 2 markers for the different labels

    Args:
        x: input data
        y: data labels

    """
    x1 = x[:, 1]
    x2 = x[:, 2]
    plt.scatter(x1[y[:, 0] == 1], x2[y[:, 0] == 1], marker='+')
    plt.scatter(x1[y[:, 0] == 0], x2[y[:, 0] == 0], marker='3')
    plt.xlabel("Exam 1 score")
    plt.ylabel("Exam 2 score")
    plt.legend(["Admitted", "Not admitted"])
    plt.show()


def sigmoid(z: np.array) -> np.array:
    """
    Applies a sigmoid function over the input

    Args:
        z: input array. in the Logistic Regression context should be  theta.T @ x

    Returns:
        h: sigmoid function applied over z
    """
    h = 1 / (1 + np.exp(-z))
    return h
