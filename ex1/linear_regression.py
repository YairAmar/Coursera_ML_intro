from utils import feature_normalize, compute_cost
import numpy as np


class LinearRegression:
    """ This class is made for training a model of linear regression """

    def __init__(self, n_features=2):
        """ The constructor of the LinearRegression object

        Args:
            n_features: number of features. For linear regression with N features send N (default 2)
        """
        self.theta = np.zeros((n_features, 1))

    def fit(self, x: np.array, y: np.array, iterations: int = 1500, learning_rate: float = 0.01,
            save_cost: bool = False, save_theta: bool = False, normalize_data: bool = True) -> dict:
        """ Trains the Linear-Regression model.
        the model's weights are optimized using batch gradient descent

        Args:
            x: input data
            y: target
            iterations: number of gradient descent iterations (default 1500)
            learning_rate: (default 0.01)
            save_cost: if True, fit will return a list of the cost-function's values (default True)
            save_theta: if True, fit will return a list of theta's values (default True)
            normalize_data: if True, the input data will be normalized to be ~N(0,1)  (default True)

        Returns:
            doc: dictionary containing the values of the cost function and theta through the training
        """
        if normalize_data:
            feature_normalize(x)

        m = len(y)
        cost_list = []
        theta_list = []

        for i in range(iterations):
            # documenting the values of theta and the cost-function
            if save_cost:
                cost_list.append(compute_cost(x, y, self.theta))
            if save_theta:
                theta_list.append(np.copy(self.theta))

            y_hat = x @ self.theta
            step = (y_hat - y).T @ x / m
            self.theta -= learning_rate * step.T

        doc = {"cost": cost_list, "theta": theta_list}
        return doc

    def fit_with_normal_eq(self, x: np.array, y: np.array) -> np.array:
        """ Solves the linear regression problem with a closed formula of normal equation

        Args:
            x: input data
            y: target

        Returns:
            theta: containing the linear regression parameters [theta_0,theta_1]
        """
        # used pseudo-inverse for stability
        self.theta = np.linalg.pinv(x.T @ x) @ x.T @ y
        theta = self.theta
        return theta

    def predict(self, x: np.array) -> np.array:
        """ Applies the linear regression model to return a prediction

        Args:
            x: input data

        Returns:
            y_hat: predicted values of y
        """
        y_hat = x @ self.theta
        return y_hat
