import numpy as np

from Sources.solver.adam import adam
from Sources.solver.cholesky import cholesky
from Sources.solver.steepest_gradient_descent import sgd


class NeuralNetwork():
    """
    Parameters mandatory inside the dictionary, recognized by the model are:
    seed = value 0. Used for test reproducibility
    layers = list [{neurons:[1,+inf), activation:tanh},
                ...
                {neurons:[1,+inf), activation:sigmoid}]. List where row are info for i layer
    solver = "sgd", "cholesky", "adam".
    """

    def __init__(self, settings: dict):
        # classic argument check
        if "seed" in settings:
            np.random.seed(settings["seed"])
        else:
            raise Exception("Seed not passed")

        if "layers" in settings and len(settings["layers"]) > 0:
            self.layers = settings["layers"]
        else:
            raise Exception("Topology of network error")

        if "solver" in settings and \
                (settings["solver"] == "sgd" or
                 settings["solver"] == "cholesky" or
                 settings["solver"] == "adam"):
            self.solver = settings["solver"]
        else:
            raise Exception("Activation function error")

        self.weights = {}
        for i in range(1, len(self.layers)):
            self.weights['W' + str(i)] = np.random.rand(self.layers[i]['neurons'],
                                                        self.layers[i - 1]['neurons'])
            self.weights['b' + str(i)] = np.zeros((self.layers[i]['neurons'], 1))

        print("Neural network initialized")

    def fit(self, X, labels, learning_rate: float, error: float, epochs: int, batch_size: int, shuffle:bool):
        """
        Parameters mandatory inside the fit model:
        :param X: training data where shapes match with initialization values
        :param labels: training y data where shapes match with initialization values
        :param epochs: number of epochs do to fitting
        :return:
        """

        if len(labels) != len(X):
            raise Exception("Label dimension mismatch")

        if epochs < 1:
            raise Exception("Epoch number error")

        if self.solver == "sgd":
            print("Running sgd")
            sgd(X, labels, self.weights, self.layers, learning_rate, error, epochs, batch_size, shuffle)
        elif self.solver == "adam":
            print("Running adam")
            adam()
        elif self.solver == "cholesky":
            print("Running cholesky")
            cholesky()
        else:
            raise Exception("Wrong solver choice")

    def predict(self, X):
        """
        :param X: X: Test data where is computed output
        :return:
        """
        pass

    def score(self, X, labels):
        """

        :param X: test data where is computed output
        :param labels: target output where is computed scored function
        :return:
        """
        pass

    def plot_model(self):
        """
        Show an image with the topology of the network
        :return:
        """
        for l in self.weights:
            print("Layer:", l, "weights:", self.weights[l])


# main used for test output
if __name__ == "__main__":
    print("Neural network tests")

    X = [[0, 1, 0], [0, 0, 1], [1, 0, 0], [1, 1, 0], [1, 1, 1]]
    labels = [[1], [0], [0], [1], [1]]

    nn = NeuralNetwork({'seed': 0,
                        'layers': [
                            {"neurons": len(X), "activation": "linear"},  # input only for dimension, insert linear
                            {"neurons": 3, "activation": "tanh"},
                            {"neurons": 4, "activation": "tanh"},
                            {"neurons": 1, "activation": "sigmoid"}  # output
                        ],
                        'solver': 'sgd'
                        })

    nn.plot_model()

    epochs = 5
    learning_rate = 0.001
    error = 0.00001
    batch_size = 2
    shuffle = True

    nn.fit(X, labels, learning_rate, error, epochs, batch_size, shuffle)
