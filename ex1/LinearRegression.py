from Utils import *


class LinearRegression:
    """
    This class is made for training a model of linear regression
    """

    def __init__(self):
        """
        The constructor of the LinearRegression object
        """
        self.theta = np.zeros((2,1))

    def fit(self, x, y, iterations=1500, learning_rate=0.01, save_cost=False,
            save_theta=False, normalize_data=True):
        """
        Trains the Linear-Regression model.
        the model's weights are optimized using batch gradient descent
        x1 - the input data
        y - target (expected output for the data)
        iterations - number of gradient descent iterations, defaulted to be 1500
        learning_rate -  defaulted to be 0.01
        save_cost - if True, fit will return a list of the cost-function's values
        save_theta - if True, fit will return a list of theta's values
        """
        if normalize_data:
            feature_normalize(x)

        self.theta = np.zeros((x.shape[1], 1))
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
        """
        Solves the linear regression problem with a closed formula of normal equation
        x1 - input data
        y - target
        returns theta, when it contains [theta_0,theta_1]
        """
        # used pseudo-inverse for stability
        self.theta = np.linalg.pinv(x.T @ x) @ x.T @ y
        return self.theta

    def predict(self, x):
        """
        x1 - input data
        returns a prediction using the linear model
        """
        y_hat = x @ self.theta
        return y_hat

