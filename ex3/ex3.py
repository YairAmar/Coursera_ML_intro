import sys
from logistic_regression import LogisticRegression
from utils import load_data


def multiclass_logistic_regression():
    """training a logistic regression model with 10 classes and tests it's results"""
    x, y = load_data(path=sys.argv[1])
    learning_rate = 0.075
    clf = LogisticRegression(x.shape[0], deg=1)
    clf.multi_class_fit(x, y, learning_rate, n_classes=10)
    acc = clf.accuracy(x, y)
    print("for learning_rate of " + str(learning_rate) + ": acc = " + str(acc))


if __name__ == "__main__":
    multiclass_logistic_regression()
