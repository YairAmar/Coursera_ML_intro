from linear_regression import LinearRegression
from utils import compute_cost, feature_normalize, plot_model, visualize_optimization, plot_cost, load_data
import numpy as np
import sys


def main():
    # reading and formatting the data
    x1, y1 = load_data(sys.argv[1])
    x2, y2 = load_data(sys.argv[2])

    # lets train a model
    lin_reg1 = LinearRegression(x1.shape[1])
    training_doc = lin_reg1.fit(x1, y1, save_cost=True, save_theta=True)

    # now training a model using normal equation
    lin_reg2 = LinearRegression(x1.shape[1])
    theta_norm_eq = lin_reg2.fit_with_normal_eq(x1, y1)
    plot_model(x1, y1, theta_norm_eq)

    # training a linear regression in more than 1 dimension
    lin_reg3 = LinearRegression(x2.shape[1])
    training_doc2 = lin_reg3.fit(x2, y2, save_cost=True, save_theta=True)

    # plotting the cost-function over iterations for the 1d and 2d data
    plot_cost(training_doc)
    plot_cost(training_doc2)

    # graph of the fitted-model over the train data
    plot_model(x1, y1, training_doc["theta"][-1])

    # visualizing the minimization path of the theta values
    visualize_optimization(x1, y1, training_doc, f=compute_cost)


if __name__ == "__main__":
    main()
