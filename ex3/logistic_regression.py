import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from trainer import forward
from trainer import gradient_descent
from utils import train_test_split


class LogisticRegression:

    def __init__(self, n_features: int, deg: int):
        """
        the constructor of the LogisticRegression object

        Args:
            n_features: number of features in the input data
            deg: polynomial degree of the feature space
        """
        self.poly = PolynomialFeatures(deg)
        self.theta = np.zeros((n_features, 1))

    def fit(self, x: np.ndarray, y: np.ndarray, max_iter: int = 400, llambda: float = 0.,
            learning_rate: float = 0.001) -> tuple:
        """
        Trains the logistic regression model

        Args:
            x: input data
            y: data classes
            max_iter: maximum number of iterations (default 400)
            llambda: lambda of the regularization argument (default 0.)
            learning_rate: learning rate (default 0.001)
        Returns:
            theta: weights of the logistic regression after training
            cost_list: list of cost-function's values through the training
            theta_list: list of theta's values through the training
        """
        theta, cost_list, theta_list = gradient_descent(self.theta, x, y, max_iter, llambda, learning_rate)
        self.theta = np.copy(theta)
        return theta, cost_list, theta_list

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        return a prediction over the input data, as a class

        Args:
            x: input data

        Returns:
            hypo: hypothesis prediction of the data's class
        """
        hypo = np.argmax(forward(self.theta, x), axis=0).astype(int)
        return hypo

    def accuracy(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        calculates the accuracy of the classifier

        Args:
            x: input data
            y: data classes

        Returns:
            acc: accuracy, correct classification rate in range 0-1
        """
        y_hat = self.predict(x)
        m = y_hat.shape[0]
        acc = np.sum(y_hat == y) / m
        return acc

    def multi_class_fit(self, x: np.ndarray, y: np.ndarray, learning_rate: float = 0.001, max_iter: int = 2000,
                        n_classes: int = 10) -> np.ndarray:
        """
        trains a multi-class logistic regression model by training <n_classes> models in a manner of 1 vs. all

        Args:
            x: input data
            y: data classes
            learning_rate: learning rate (default 0.001)
            max_iter: maximum number of iterations (default 2000)
            n_classes: number of classes (default 10)
        Returns:
            thetas: an array with the theta values of all of the models trained
        """
        thetas = []

        for i in range(n_classes):
            dup_y = np.copy(y)
            dup_y[y == i] = 1
            dup_y[y != i] = 0
            x_train, y_train, x_test, y_test = train_test_split(x, dup_y)
            clf = LogisticRegression(x.shape[1], deg=1)
            curr_theta, _, _ = clf.fit(x_train, y_train, learning_rate=learning_rate, max_iter=max_iter)
            thetas.append(np.copy(curr_theta))

        thetas = np.array(thetas)
        self.theta = np.copy(thetas)
        return thetas
