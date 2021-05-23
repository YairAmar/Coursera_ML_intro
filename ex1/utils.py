import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D
from typing import Dict, Callable


def compute_cost(x: np.array, y: np.array, theta: np.array) -> np.float:
    """
    Computes the cost function's value

    Args:
        x: input data, with a column of 1
        y: input target
        theta: weights of the linear regression

    Returns:
        cost: the value of the cost function
    """
    m = len(y)
    y_hat = x @ theta
    cost = ((y_hat - y).T @ (y_hat - y) / (2 * m)).item()
    return cost


def feature_normalize(x: np.array):
    """
    Normalizes data to be ~ N(0,1)

    Args:
        x: input data
    """
    def normalize(vec): return (vec - np.mean(vec)) / np.std(vec)
    for i in np.arange(x.shape[1] - 1) + 1:
        x[:, i] = normalize(x[:, i])


def plot_model(x: np.array, y: np.array, theta: np.array):
    """ Visualizes a trained regression model, while plotting the data as well
    Works only for 1-d data

    Args:
        x: the input data
        y: target
        theta: weights of the linear regression
    """
    plt.scatter(x[:, -1], y, marker="3")
    plt.xlabel("x1")
    plt.ylabel("y")
    plt.title("fitted model")
    plt.grid()
    t = np.linspace(np.min(x[:, -1]), np.max(x[:, -1]), 10000)
    line = theta[0] + t * theta[1]
    plt.plot(t, line, 'r')
    plt.show()


def visualize_optimization(x: np.array, y: np.array, doc_dict: Dict, f: Callable):
    """
    Creates a 3-d surface plot for the given cost-function, with the values that theta got through the training
    Works only for 1-d data

    Args:
        x: input data
        y: target
        doc_dict: output of the fit function in LinearRegression class
        f: function that computes the wanted cost function
    """
    fig = plt.figure(figsize=(12, 12))
    ax = fig.gca(projection='3d')

    theta0s = np.arange(-10, 10, .5)
    theta1s = np.arange(-1, 8, .5)
    th0, th1, cost = [], [], []
    for theta0 in theta0s:
        for theta1 in theta1s:
            th0.append(theta0)
            th1.append(theta1)
            tmp_theta = np.array([[theta0], [theta1]])
            cost.append(f(x, y, tmp_theta))

    ax.scatter(th0, th1, cost, c=np.abs(cost))
    plt.xlabel(r'$\theta_0$', fontsize=30)
    plt.ylabel(r'$\theta_1$', fontsize=30)
    plt.title('Cost', fontsize=20)

    theta_hist = doc_dict["theta"]
    cost_hist = doc_dict["cost"]

    th_hist0 = [th[0] for th in theta_hist]
    th_hist1 = [th[1] for th in theta_hist]
    ax.plot(th_hist0, th_hist1, cost_hist, 'ro-')
    plt.show()


def plot_cost(doc: Dict):
    """
    Plots the cost function over training iterations

    Args:
        doc: output of the fit function in LinearRegression class
        a dictionary with the key "cost", and it's fitting values.
    """
    plt.plot(doc["cost"])
    plt.xlabel("iterations (dozens)")
    plt.ylabel("cost function")
    plt.xlim((0, 1500))
    plt.title("cost function over iterations")
    plt.show()


def load_data(path: str) -> tuple:
    """
    Loading data and formatting for the latter linear regression

    Args:
        path: directory path of the csv file containing the data

    Returns:
        x: an array with the data
        y: target values
    """
    print(path)
    data1 = np.loadtxt(path, delimiter=',', unpack=True)
    x = np.array(data1[:-1]).T
    y = np.array(data1[-1:]).T
    x = np.insert(x, 0, 1, axis=1)
    return x, y
