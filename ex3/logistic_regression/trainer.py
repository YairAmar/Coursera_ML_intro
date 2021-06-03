import numpy as np
from utils import sigmoid


def forward(theta: np.ndarray, x: np.ndarray) -> np.ndarray:
    """
    computes the hypothesis function

    Args:
        theta: logistic regression's parameters
        x: input data

    Returns:
        hypothesis: sigmoid function applied over z=x@theta
    """
    hypothesis = sigmoid(x @ theta)
    return hypothesis


def compute_gradient(theta: np.ndarray, x: np.ndarray, y: np.ndarray, llambda: float = 0.) -> np.ndarray:
    """
    Computes the gradient -J(theta) for the weight-vector theta, with given data (x,y).
    takes into consideration the regularization argument as well.

    Args:
        theta: theta vector of the model
        x: input data
        y: data classes
        llambda: lambda of the regularization argument (default 0.)

    Returns:
        grad: gradient vector - gradient value for each weight of the logistic regression
    """
    m = len(y)
    reg_term = llambda * theta[1:] / m
    cost_term = x.T @ (forward(theta, x) - y) / m
    grad = cost_term
    grad[1:] += reg_term
    return grad


def cost_function(theta: np.ndarray, x: np.ndarray, y: np.ndarray, llambda: float = 0.) -> float:
    """
    calculates the cost function with given regularization argument

    Args:
        theta: theta vector of the model
        x: input data
        y: data classes
        llambda: lambda of the regularization argument (default 0.)

    Returns:
        cost: cost function's value
    """
    m = len(y)
    hypo = forward(theta, x)
    log_h = np.log(hypo)
    cost_term = (-y.T @ log_h - (1 - y).T @ (np.log(1 - hypo)))  # -y*log(h)-(1-y)log(1-h)
    reg_term = (llambda / 2) * (theta[1:].T @ theta[1:])  # no need for theta[0]
    cost = ((cost_term + reg_term) / m).item()
    return cost


def gradient_descent(theta: np.ndarray, x: np.ndarray, y: np.ndarray, max_iter: int = 400, llambda: float = 0.,
                     learning_rate: float = 0.001) -> tuple:
    """
    Finding the values of theta that minimize the cost function

    Args:
        theta: theta vector of the model
        x: input data
        y: data classes
        max_iter: maximum number of iterations (default 400)
        llambda: lambda of the regularization argument (default 0.)
        learning_rate:

    Returns:
        theta: theta vector of the model, after optimization
        theta_list: list of all theta-values acquire through the training
        cost_list: list of all cost-function values acquire through the training
    """
    theta = np.copy(theta)  # in order to not override the theta in the outer context
    cost_list = []
    theta_list = []
    for i in range(max_iter):
        cost_list.append(cost_function(theta=theta, x=x, y=y))
        curr_theta = np.copy(theta)
        theta_list.append(np.copy(curr_theta))
        theta -= learning_rate * compute_gradient(curr_theta, x, y, llambda)

    return theta, theta_list, cost_list
