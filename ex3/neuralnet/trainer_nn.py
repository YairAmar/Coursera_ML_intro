import numpy as np
from utils import sigmoid


def forward(weights: list, x: np.ndarray) -> np.ndarray:
    """
    Applies feed-forward on the given neural network

    Args:
        weights: weights of the neural-net, layer by layer
        x: input data

    Returns:
        probability of each class
    """
    a = x.T
    for layer in weights:
        a = np.insert(a, 0, 1, axis=0)
        z = layer @ a
        a = sigmoid(z)

    return a
