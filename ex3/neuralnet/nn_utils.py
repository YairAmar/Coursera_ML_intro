from scipy.io import loadmat


def load_nn_weights(path: str) -> tuple:
    """
    Loading weights of a pre-trained NN
    Args:
        path: path to the .mat file

    Returns:
        theta1: weights of the 1st layer
        theta2: weights of the 2nd layer
    """
    weights = loadmat(path)
    theta1 = weights["Theta1"]
    theta2 = weights["Theta2"]
    return theta1, theta2
