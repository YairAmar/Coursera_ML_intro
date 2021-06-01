import numpy as np


class NeuralNetwork:
    def __init__(self):
        """The constructor of the NN-model object"""
        self.thetas = load_nn_weights(sys.argv[2])

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Applies a prediction of the neural-net over x
        Args:
            x: input data

        Returns:
            prediction: predicted label for each data-point
        """
        prediction = np.argmax(forward(self.thetas, x), axis=0).astype(int)
        return prediction