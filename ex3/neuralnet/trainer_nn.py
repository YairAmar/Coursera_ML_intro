import numpy as np
from utils import sigmoid


def forward(theta1: np.ndarray, theta2: np.ndarray, x: np.ndarray) -> np.ndarray:
    """
    Applies feed-forward on the given neural network

    Args:
        theta1: weights of the 1st layer in the neural-net
        theta2: weights of the 2nd layer in the neural-net
        x: input data

    Returns:
        probability of each class
    """
    a1 = np.insert(x.T, 0, 1, axis=0)
    z2 = theta1 @ a1
    a2 = sigmoid(z2)
    a2 = np.insert(a2, 0, 1, axis=0)
    z3 = theta2 @ a2
    a3 = sigmoid(z3)
    return a3
