import numpy as np
from utils import h, cost_function, plot_data
from scipy import optimize
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt


class LogisticRegression:

    def __init__(self, n_features: int, deg: int):
        """
        the constructor of the LogisticRegression object

        Args:
            n_features: number of features in the input data
        """
        self.poly = PolynomialFeatures(deg)
        self.theta = np.zeros((n_features, 1))

    def fit(self, x: np.array, y: np.array, max_iter: int = 400, llambda: float = 0.,
            method: str = "Nelder-Mead") -> tuple:
        """
        Trains the logistic regression model

        Args:
            x: input data
            y: target
            max_iter: maximum number of iterations (default 400)
            llambda: lambda of the regularization argument (default 0.)
            method: optimization method (default "Nelder-Mead")
                    for non-linear polynomials use "BFGS"
        Returns:
            theta: weights of the logistic regression after training
            cost_fun: value of the cost function after training
        """
        res = optimize.minimize(cost_function, x0=self.theta, args=(x, y, llambda), method=method,
                                options={"maxiter": max_iter, "disp": False})
        self.theta = np.array([res.x])
        cost_fun = res.fun
        theta = np.copy(self.theta)
        return theta, cost_fun

    def plot_decision_bounds(self, x: np.array, y: np.array, llambda: float = 0.):
        """
        Plots the decision boundaries of the model

        Args:
            x: input data
            y: target
            llambda: lambda of the regularization argument (default 0.)
        """
        boundary_x1 = np.array([np.min(x[:, 1]), np.max(x[:, 1])])
        boundary_x2 = np.array([np.min(x[:, 2]), np.max(x[:, 2])])
        x1_grid = np.arange(boundary_x1[0], boundary_x1[1], 0.1)
        x2_grid = np.arange(boundary_x2[0], boundary_x2[1], 0.1)
        z_vals = np.zeros((len(x1_grid), len(x2_grid)))
        for x1_ind, x1_val in enumerate(x1_grid):
            for x2_ind, x2_val in enumerate(x2_grid):
                sample = np.array([x1_val, x2_val]).reshape(1, -1)
                poly_features = self.poly.fit_transform(sample).reshape(1, -1)
                z_vals[x1_ind][x2_ind] = poly_features @ self.theta.T

        z_vals = z_vals.T
        cont = plt.contour(x1_grid, x2_grid, z_vals, [0])
        fmt = {0: 'Lambda = %d' % llambda}
        plt.clabel(cont, inline=1, fontsize=15, fmt=fmt)
        plt.title("Decision Boundary")
        plot_data(x[:, 1:], y)

    def predict(self, x: np.array) -> np.array:
        """
        return a prediction over the input data, as a class

        Args:
            x: input data

        Returns:
            hypo: hypothesis prediction of the data's class
        """
        hypo = h(self.theta, x) >= 0.5
        return hypo

    def accuracy(self, x: np.array, y: np.array) -> float:
        """
        calculates the accuracy of the classifier

        Args:
            x: input data
            y: target

        Returns:
            acc: accuracy, correct classification rate in range 0-1
        """
        pred = self.predict(x).astype(int) - y
        m = pred.shape[0]
        acc = 1 - np.sum(np.abs(pred)) / m
        return acc
