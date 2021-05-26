from utils import load_data, plot_data
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
    log_reg = LogisticRegression(x.shape[1], deg=6)

    plt.figure(figsize=(12, 16))
    # Lambda = 0
    log_reg.fit(x, y, max_iter=400, llambda=0., method="BFGS")
    plt.subplot(221)
    log_reg.plot_decision_bounds(x, y, llambda=0.)
    # Lambda = 1
    log_reg.fit(x, y, max_iter=400, llambda=1., method="BFGS")
    plt.subplot(222)
    log_reg.plot_decision_bounds(x, y, llambda=1.)
    # Lambda = 10
    log_reg.fit(x, y, max_iter=400, llambda=10., method="BFGS")
    plt.subplot(223)
    log_reg.plot_decision_bounds(x, y, llambda=10.)
    # Lambda = 100.
    log_reg.fit(x, y, max_iter=400, llambda=100., method="BFGS")
    plt.subplot(224)
    log_reg.plot_decision_bounds(x, y, llambda=100.)
    plt.show()


def main():
    x1, y1 = load_data(sys.argv[1], add_bias=False)
    plot_data(x1, y1)
    plt.show()
    poly = PolynomialFeatures(1)
    x1 = poly.fit_transform(x1)
    log_reg1 = LogisticRegression(x1.shape[1], deg=1)
    log_reg1.fit(x1, y1, max_iter=400, llambda=0.)
    log_reg1.plot_decision_bounds(x1, y1)
    plt.show()
    print("accuracy = ", log_reg1.accuracy(x1, y1))
    x2, y2 = load_data(sys.argv[2], add_bias=False)
    plt.grid()
    plot_data(x2, y2)
    plt.show()
    plot_w_regularization(x2, y2)


if __name__ == "__main__":
    main()
