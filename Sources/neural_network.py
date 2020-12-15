import numpy as np


class NeuralNetwork():
    """
    Parameters mandatory inside the dictionary, recognized by the model are:
    seed = value 0. Used for test reproducibility
    units = list [3,2,1]. Each value corresponds to the number of units in i layer
    activation = string tanh, sigmoid. Activation function chosen
    features = value [1, +inf). Number of column each pattern has
    """

    def __init__(self, settings: dict):
        # classic argument check
        if "seed" in settings:
            np.random.seed(settings["seed"])
        else:
            raise Exception("Seed not passed")

        if "units" in settings and len(settings["units"]) > 0:
            self.units = settings["units"]
        else:
            raise Exception("Neurons number error")

        if "activation" in settings and (settings["activation"] == "tanh" or settings["activation"] == "sigmoid"):
            self.activation = settings["activation"]
        else:
            raise Exception("Activation function error")

        if "features" in settings and settings["features"] > 0:
            self.features = settings["features"]
        else:
            raise Exception("Column number of pattern error")

        self.weights = []
        # add input layer: row are neurons (1 = neuron 1) and column the corresponding weight for the i neuron
        self.weights.append({"W1": np.random.rand(self.units[0], self.features), "b1": np.zeros(self.units[0])})

        # add hidden layers: row are neurons (1 = neuron 1) and column the corresponding weight for the i neuron
        for i in range(1, len(self.units) - 1):
            self.weights.append([{'W' + str(i + 1): [np.random.rand(self.units[i], self.units[i - 1])],
                                  "b" + str(i + 1): np.zeros(self.units[i])}])

        # add output layer: each row is one output with corresponding weights
        self.weights.append([{'Wout': [np.random.rand(self.units[-1], self.units[len(self.units) - 2])],
                              "bout": np.zeros(self.units[-1])}])

        print("Neural network initialized")

    def fit(self, X, labels):
        """
        Parameters mandatory inside the fit model:
        :param X: training data where shapes match with initialization values
        :param labels: training y data where shapes match with initialization values
        :return:
        """
        # classic argument check
        for i in range(0, len(X[0])):  # for each samples check features dimension
            if len(X[i][1]) != self.features:
                raise Exception("Miss features in some pattern")

        if len(labels) != len(X[0]):
            raise Exception("Label dimension mismatch")

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
            print(l)


# main used for test output
if __name__ == "__main__":
    print("Neural network tests")

    nn = NeuralNetwork({'seed': 0,
                        'units': [3, 1],
                        'activation': 'sigmoid',
                        'features': 4})

    nn.plot_model()