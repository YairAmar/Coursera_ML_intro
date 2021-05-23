from utils import feature_normalize, compute_cost
import numpy as np


class LinearRegression:
    """ This class is made for training a model of linear regression """

    def __init__(self, n_features):
        """ The constructor of the LinearRegression object

        Keyword arguments:
        n_features - number of features. For linear regression with N features send N
        """
        self.theta = np.zeros((n_features, 1))

    def fit(self, x, y, iterations=1500, learning_rate=0.01, save_cost=False,
            save_theta=False, normalize_data=True):
        """ Trains the Linear-Regression model.
        the model's weights are optimized using batch gradient descent

        Keyword arguments:
        x -- input data
        y -- target
        iterations -- number of gradient descent iterations (default 1500)
        learning_rate -  (default 0.01)
        save_cost - if True, fit will return a list of the cost-function's values (default False)
        save_theta - if True, fit will return a list of theta's values (default False)
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

        return {"cost": cost_list, "theta": theta_list}

    def fit_with_normal_eq(self, x, y):
        """ Solves the linear regression problem with a closed formula of normal equation
        returns theta, when it contains [theta_0,theta_1]

        Keyword arguments:
        x - input data
        y - target
        """
        # used pseudo-inverse for stability
        self.theta = np.linalg.pinv(x.T @ x) @ x.T @ y
        return self.theta

    def predict(self, x):
        """ Applies the linear regression model to return a prediction

        Keyword arguments:
        x - input data
        """
        y_hat = x @ self.theta
        return y_hat
