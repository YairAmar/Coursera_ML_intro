import sys
import numpy as np
import matplotlib.pyplot as plt
from utils import pre_process
from logistic_regression import LogisticRegression


def plot_w_regularization(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, y_test: np.ndarray):
    """
    Plots 4 graphs of the logistic regression model with polynomial features of degree 6
    and different values of regularization parameter in the cost function

    Args:
        x_train: train data
        y_train: train labels
        x_test: test data
        y_test: test labels
    """
    log_reg = LogisticRegression(x_train.shape[1], deg=6)

    lambda_values = [0., 1., 10., 100.]
    plt.figure(figsize=(12, 16))

    for i, llambda in enumerate(lambda_values):
        log_reg.fit(x_train, y_train, max_iter=400, llambda=llambda, method="BFGS")
        plt.subplot(2, 2, i+1)
        log_reg.plot_decision_bounds(x_train, y_train, llambda=llambda)
        train_accuracy = log_reg.compute_accuracy(x_train, y_train)
        test_accuracy = log_reg.compute_accuracy(x_test, y_test)
        print(f"for lambda = {llambda}, train accuracy = {train_accuracy}")
        print(f"for lambda = {llambda}, test accuracy = {test_accuracy}")

    plt.show()


def main():
    x1_train, y1_train, x1_test, y1_test = pre_process(file_path=sys.argv[1])
    log_reg1 = LogisticRegression(x1_train.shape[1], deg=1)
    log_reg1.fit(x1_train, y1_train, max_iter=400, llambda=0.)
    train_accuracy = log_reg1.compute_accuracy(x1_train, y1_train)
    test_accuracy = log_reg1.compute_accuracy(x1_test, y1_test)
    print(f"the train accuracy of the linear model is: {train_accuracy}")
    print(f"the test accuracy of the linear model is: {test_accuracy}")

    log_reg1.plot_decision_bounds(x1_train, y1_train)
    plt.show()
    x2_train, y2_train, x2_test, y2_test = pre_process(file_path=sys.argv[2], poly_deg=6)
    plot_w_regularization(x2_train, y2_train, x2_test, y2_test)


if __name__ == "__main__":
    main()
