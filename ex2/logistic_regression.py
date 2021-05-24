import numpy as np
from utils import sigmoid

class LogisticRegression:

    def __init__(self, n_features: int):
        """
        the constructor of the LogisticRegression object

        Args:
            n_features: number of features in the input data
        """
        # its n_features + 1 since we need a theta_0 too
        self.theta = np.zeros((n_features+1, 1))

    def fit(self, x: np.array, y: np.array, regularization: bool = False):
        """

        Args:
            x:
            y:
            regularization:

        Returns:

        """
        z = self.theta.T @ x
        h = 1 / (1 + np.exp(-1))
        cost = -y * np.log(h) - (1-y) * np.log(h)


    def predict(self, x: np.array) -> np.array:
        """

        Args:
            x:

        Returns:

        """
        # TODO - pick a threshold. meanwhile we work with a threshold of 0.5
        z = self.theta.T @ x
        h = sigmoid(z)
        return h
