import matplotlib as plt

from Sources.solver.adam import adam
from Sources.solver.cholesky import cholesky
from Sources.solver.steepest_gradient_descent import sgd
from Sources.tools.activation_function import *
from Sources.tools.score_function import *
from Sources.tools.preprocessing import one_hot
from Sources.tools.load_dataset import *
from Sources.tools.weight_initialization import xavier_init


class NeuralNetwork:
    """
    Parameters mandatory inside the dictionary, recognized by the model are:
    seed = value 0. Used for test reproducibility
    layers = list [{neurons:[1,+inf), activation:tanh},
                ...
                {neurons:[1,+inf), activation:sigmoid}]. List where row are info for i layer
    solver = "sgd", "cholesky", "adam"
    problem = "classification", "regression"
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
                 settings["solver"] == "adam" or
                 settings["solver"] == "cholesky"):
            self.solver = settings["solver"]
        else:
            raise Exception("Activation function error")

        if "problem" in settings and settings["problem"] == "classification" or settings["problem"] == "regression":
            self.problem = settings["problem"]
        else:
            raise Exception("Problem statement error")

        self.__initialize_weights()

        print("Neural network initialized")

    def __initialize_weights(self):
        self.weights = {}

        for i in range(1, len(self.layers)):
            self.weights['W' + str(i)] = xavier_init((self.layers[i - 1]["neurons"], self.layers[i]["neurons"]),
                                                     self.layers[0]["neurons"], self.layers[-1]["neurons"])

            self.weights['b' + str(i)] = np.zeros((1, self.layers[i]['neurons']))

    def fit(self, X, labels, X_validation, labels_validation, hyperparameters: dict, epochs=1, batch_size=32, shuffle=True):
        """
        :param X:
        :param labels:
        :param hyperparameters:
        :param epochs:
        :param batch_size:
        :param shuffle:
        :return:
        """

        if len(labels) != len(X):
            raise Exception("Label dimension mismatch")

        if epochs < 1:
            raise Exception("Epoch number error")

        if self.solver == "sgd":
            print("\nRunning sgd")
            self.history = sgd(X, labels, self.weights, self.layers, hyperparameters, epochs, batch_size, shuffle,
                               X_validation, labels_validation)

        elif self.solver == "adam":
            print("\nRunning adam")
            self.history = adam(X, labels, self.weights, self.layers, hyperparameters, epochs, batch_size, shuffle,
                                X_validation, labels_validation)

        elif self.solver == "cholesky":
            print("\nRunning cholesky")

            if self.layers[-1]["activation"] != "linear":
                raise Exception("Per cholesky l'ultima funzione di attivazione deve mecessariamente essere lineare")

            cholesky(X, labels, hyperparameters["lambda"], self.weights, self.layers)

        else:
            raise Exception("Wrong solver choice")

    def predict(self, X):
        """
        :param X: X: Test data where is computed output
        :return:
        """

        output = np.array(X)

        for i in range(1, len(self.layers)):

            output = output @ self.weights['W' + str(i)] + self.weights['b' + str(i)]

            if self.layers[i]["activation"] == "sigmoid":
                output = sigmoid(output)
            if self.layers[i]["activation"] == "tanh":
                output = tanh(output)

        return output

    def score(self, X, labels):
        """
        :param X: test data where is computed output
        :param labels: target output where is computed scored function
        :return: mean square error over the test set
        """

        output = np.array(X)

        for i in range(1, len(self.layers)):

            if i != len(self.layers):
                output = output @ self.weights['W' + str(i)] + self.weights['b' + str(i)]

            if self.layers[i]["activation"] == "sigmoid":
                output = sigmoid(output)
            if self.layers[i]["activation"] == "tanh":
                output = tanh(output)

        return mean_squared_error(output, labels)

    def plot_model(self):
        """
        Show an image with the topology of the network
        :return:
        """
        for keys in self.weights:
            print("-->", keys, ":")
            for value in self.weights[keys]:
                print(value)

    def plot_graph(self):
        plt.plot(self.history['acc_train'])
        plt.plot(self.history['acc_validation'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper left')
        plt.show()
        # "Loss"
        plt.plot(self.history['mse_train'])
        plt.plot(self.history['mse_validation'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper left')
        plt.show()
