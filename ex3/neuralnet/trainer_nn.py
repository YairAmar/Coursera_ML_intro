import numpy as np
from utils import sigmoid


def forward(theta_list: np.ndarray, x: np.ndarray) -> np.ndarray:
    """
    Applies feed-forward on the given neural network

    Args:
        theta_list: weights of all of the layers in the neural-net, one by one
        x: input data

    Returns:
        probability of each class
    """
    a = x.T

    for theta in theta_list:
        a = np.insert(a, 0, 1, axis=0)
        z = theta @ a
        a = sigmoid(z)

    return a
