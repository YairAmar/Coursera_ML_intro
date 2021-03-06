import sys
import numpy as np
from neuralnet.trainer_nn import forward
from neuralnet.nn_utils import load_nn_weights


class NeuralNet:

    def __init__(self):
        """The constructor of the NN-model object"""
        self.weights = load_nn_weights(sys.argv[2])

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Applies a prediction of the neural-net over x
        Args:
            x: input data

        Returns:
            prediction: predicted label for each data-point
        """
        prediction = np.argmax(forward(self.weights, x), axis=0).astype(int)
        return prediction
