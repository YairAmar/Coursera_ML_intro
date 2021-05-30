import sys
import numpy as np
import matplotlib.pyplot as plt
from utils import load_data, plot_data, train_test_split
from logistic_regression import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures


def plot_w_regularization(x: np.ndarray, y: np.ndarray):
    """
    Plots 4 graphs of the logistic regression model with polynomial features of degree 6
    and different values of regularization parameter in the cost function

    Args:
        x: input data
        y: target
    """
    poly_reg = PolynomialFeatures(6)
    x = poly_reg.fit_transform(x)
    x_train, y_train, x_test, y_test = train_test_split(x, y)
    log_reg = LogisticRegression(x_train.shape[1], deg=6)

    llambdas = [0., 1., 10., 100.]
    plt.figure(figsize=(12, 16))

    for i, llambda in enumerate(llambdas):
        log_reg.fit(x_train, y_train, max_iter=400, llambda=llambda, method="BFGS")
        plt.subplot(2, 2, i+1)
        log_reg.plot_decision_bounds(x_train, y_train, llambda=llambda)
        train_accuracy = log_reg.compute_accuracy(x_train, y_train)
        test_accuracy = log_reg.compute_accuracy(x_test, y_test)
        print(f"for lambda = {llambda}, train accuracy = {train_accuracy}")
        print(f"for lambda = {llambda}, test accuracy = {test_accuracy}")

    plt.show()


def main():
    x1, y1 = load_data(sys.argv[1], add_bias=False)
    plot_data(x1, y1)
    plt.show()
    poly = PolynomialFeatures(1)
    x1 = poly.fit_transform(x1)
    x1_train, y1_train, x1_test, y1_test = train_test_split(x1, y1, ratio=0.8)

    log_reg1 = LogisticRegression(x1_train.shape[1], deg=1)
    log_reg1.fit(x1_train, y1_train, max_iter=400, llambda=0.)
    train_accuracy = log_reg1.compute_accuracy(x1_train, y1_train)
    test_accuracy = log_reg1.compute_accuracy(x1_test, y1_test)
    print(f"the train accuracy of the linear model is: {train_accuracy}")
    print(f"the test accuracy of the linear model is: {test_accuracy}")

    log_reg1.plot_decision_bounds(x1, y1)
    plt.show()
    x2, y2 = load_data(sys.argv[2], add_bias=False)
    plt.grid()
    plot_data(x2, y2)
    plt.show()
    plot_w_regularization(x2, y2)


if __name__ == "__main__":
    main()
