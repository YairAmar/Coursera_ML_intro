from LinearRegression import *
from Utils import *


if __name__ == "__main__":
    # loading data
    data_dir1 = "data\ex1data1.txt"
    data_dir2 = "data\ex1data2.txt"
    data1 = np.loadtxt(data_dir1, delimiter=',', unpack=True)
    data2 = np.loadtxt(data_dir2, delimiter=',', unpack=True)
    x1 = np.array(data1[:-1]).T
    y1 = np.array(data1[-1:]).T
    x1 = np.insert(x1, 0, 1, axis=1)
    x2 = np.array(data2[:-1]).T
    y2 = np.array(data2[-1:]).T
    x2 = np.insert(x2, 0, 1, axis=1)

    # lets train a model
    lin_reg1 = LinearRegression()
    training_doc = lin_reg1.fit(x1, y1, save_cost=True, save_theta=True)

    # now training a model using normal equation
    lin_reg2 = LinearRegression()
    theta_norm_eq = lin_reg2.fit_with_normal_eq(x1, y1)
    plot_model(x1, y1, theta_norm_eq)

    # training a linear regression in more than 1 dimension
    lin_reg3 = LinearRegression()
    training_doc2 = lin_reg3.fit(x2, y2, save_cost=True, save_theta=True)

    # plotting the cost-function over iterations for the 1d and 2d data
    plot_cost(training_doc)
    plot_cost(training_doc2)

    # graph of the fitted-model over the train data
    plot_model(x1, y1, training_doc["theta"][-1])

    # visualizing the minimization path of the theta values
    visualize_optimization(x1, y1, training_doc, f=compute_cost)


