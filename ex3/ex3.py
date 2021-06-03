import sys
from logistic_regression.logistic_regression import LogisticRegression
from utils import load_data, compute_accuracy
from neuralnet.neural_net import NeuralNet


def multiclass_logistic_regression():
    """Training a logistic regression model with 10 classes and tests it's results"""
    x, y = load_data(path=sys.argv[1])
    learning_rate = 0.075
    clf = LogisticRegression(x.shape[0], deg=1)
    clf.multi_class_fit(x, y, learning_rate, n_classes=10)
    acc = compute_accuracy(clf, x, y)
    print(f"For logistic regression, with learning_rate of {learning_rate}: accuracy = {acc*100}%")


def neural_network_classification():
    """Evaluating a pre trained neural network classifier"""
    x, y = load_data(path=sys.argv[1], add_bias=False)
    # weird scaling specific for the way they trained their model (matlab...)
    y[y == 0] = 10
    y -= 1

    clf = NeuralNet()
    acc = compute_accuracy(clf, x, y)
    print(f"The neural-net accuracy is: {acc*100}%")


if __name__ == "__main__":
    multiclass_logistic_regression()
    neural_network_classification()
