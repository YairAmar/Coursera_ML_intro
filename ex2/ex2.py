from utils import load_data, plot_data, train_test_split
import sys
from logistic_regression import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt


def plot_w_regularization(x, y):
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

    plt.figure(figsize=(12, 16))
    # Lambda = 0
    log_reg.fit(x_train, y_train, max_iter=400, llambda=0., method="BFGS")
    plt.subplot(221)
    log_reg.plot_decision_bounds(x_train, y_train, llambda=0.)
    print("for lambda = 0, train accuracy = ", log_reg.accuracy(x_train, y_train))
    print("for lambda = 0, test accuracy = ", log_reg.accuracy(x_test, y_test))
    # Lambda = 1
    log_reg.fit(x_train, y_train, max_iter=400, llambda=1., method="BFGS")
    plt.subplot(222)
    log_reg.plot_decision_bounds(x_train, y_train, llambda=1.)
    print("for lambda = 1, train accuracy = ", log_reg.accuracy(x_train, y_train))
    print("for lambda = 1, test accuracy = ", log_reg.accuracy(x_test, y_test))

    # Lambda = 10
    log_reg.fit(x_train, y_train, max_iter=400, llambda=10., method="BFGS")
    plt.subplot(223)
    log_reg.plot_decision_bounds(x_train, y_train, llambda=10.)
    print("for lambda = 10, train accuracy = ", log_reg.accuracy(x_train, y_train))
    print("for lambda = 10, test accuracy = ", log_reg.accuracy(x_test, y_test))

    # Lambda = 100.
    log_reg.fit(x_train, y_train, max_iter=400, llambda=100., method="BFGS")
    plt.subplot(224)
    log_reg.plot_decision_bounds(x_train, y_train, llambda=100.)
    print("for lambda = 100, train accuracy = ", log_reg.accuracy(x_train, y_train))
    print("for lambda = 100, test accuracy = ", log_reg.accuracy(x_test, y_test))
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
    print("the train accuracy of the linear model is: ", log_reg1.accuracy(x1_train, y1_train))
    print("the test accuracy of the linear model is: ", log_reg1.accuracy(x1_test, y1_test))

    log_reg1.plot_decision_bounds(x1, y1)
    plt.show()
    x2, y2 = load_data(sys.argv[2], add_bias=False)
    plt.grid()
    plot_data(x2, y2)
    plt.show()
    plot_w_regularization(x2, y2)


if __name__ == "__main__":
    main()
