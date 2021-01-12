import numpy as np

from Sources.solver.adam import adam
from Sources.solver.cholesky import cholesky
from Sources.solver.extreme_adam import extreme_adam
from Sources.solver.steepest_gradient_descent import sgd
from Sources.tools.activation_function import *
from Sources.tools.score_function import *
from Sources.tools.preprocessing import one_hot
from Sources.tools.load_dataset import *
from Sources.tools.weight_initialization import xavier_init, uniform_weights
import matplotlib.pyplot as plt


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
                 settings["solver"] == "extreme_adam" or
                 settings["solver"] == "cholesky"):
            self.solver = settings["solver"]
        else:
            raise Exception("Activation function error")

        if "problem" in settings and settings["problem"] == "classification" or settings["problem"] == "regression":
            self.problem = settings["problem"]
        else:
            raise Exception("Problem statement error")

        if "initialization" in settings and settings["initialization"] == "uniform" or settings["initialization"] == "xavier":
            self.initialization = settings["initialization"]
        else:
            raise Exception("Initialization statemenet error")

        self.__initialize_weights()


    def __initialize_weights(self):
        self.weights = {}
        if self.initialization == "uniform":
            for i in range(1, len(self.layers)):
                self.weights['W' + str(i)] = uniform_weights((self.layers[i - 1]["neurons"], self.layers[i]["neurons"]))

                self.weights['b' + str(i)] = np.zeros((1, self.layers[i]['neurons']))
        elif self.initialization == "xavier":
            for i in range(1, len(self.layers)):
                self.weights['W' + str(i)] = xavier_init((self.layers[i - 1]["neurons"], self.layers[i]["neurons"]))

                self.weights['b' + str(i)] = np.zeros((1, self.layers[i]['neurons']))

    def fit(self, X, labels, X_validation, labels_validation, hyperparameters: dict, epochs=1, batch_size=32,
            shuffle=True):
        """
        :param labels_validation:
        :param X_validation:
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
            print("Running sgd")
            self.history = sgd(X, labels, self, hyperparameters, epochs, batch_size, shuffle,
                               X_validation, labels_validation)

        elif self.solver == "adam":
            print("Running adam")
            self.history = adam(X, labels, self, hyperparameters, epochs, batch_size, shuffle,
                                X_validation, labels_validation)

        elif self.solver == "extreme_adam":
            print("Running extreme adam")
            self.history = extreme_adam(X, labels, self, hyperparameters, epochs, batch_size, shuffle,
                                        X_validation, labels_validation)

        elif self.solver == "cholesky":
            print("Running cholesky")

            if self.layers[-1]["activation"] != "linear":
                raise Exception("Per cholesky l'ultima funzione di attivazione deve mecessariamente essere lineare")

            cholesky(X, labels, hyperparameters["lambda"], self.weights, self.layers)

        else:
            raise Exception("Wrong solver choice")

    def predict(self, X):
        """
        Compute the output of the network given examples
        :param X: X: Test data where is computed output
        :return:
        """

        output = np.array(X)

        for i in range(1, len(self.layers)):
            output = output @ self.weights['W' + str(i)] + self.weights['b' + str(i)]

            output = apply_activation(self.layers[i]["activation"], output)

        return output

    def score_mse(self, X, labels):
        """
        Compute mse of the network given inputs data
        :param X: test data where is computed output
        :param labels: target output where is computed scored function
        :return: mean square error over the test set
        """

        return mean_squared_error(self.predict(X), labels)

    def score_accuracy(self, X, labels):
        """
        COmpute accuracy of the model given inputs data
        :param X: test data where is computed accuracy
        :param labels:
        :return:
        """
        X = self.predict(X)
        treshold_list = [[1] if i > 0.5 else [0] for i in X]

        return classification_accuracy(treshold_list, labels)

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
        # "Loss"
        fontsize_legend_axis = 12
        plt.plot(self.history['mse_train'])
        plt.plot(self.history['mse_validation'], linestyle="--")
        plt.title('Model MSE')
        plt.ylabel('Mean squared error')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper right', fontsize=fontsize_legend_axis)
        plt.xticks(fontsize=fontsize_legend_axis)
        plt.yticks(fontsize=fontsize_legend_axis)
        plt.grid()
        plt.show()
        if self.problem != "regression":
            plt.plot(self.history['acc_train'])
            plt.plot(self.history['acc_validation'], linestyle="--")
            plt.title('Model ACC')
            plt.ylabel('Accuracy')
            plt.xlabel('Epoch')
            plt.legend(['Train', 'Validation'], loc='lower right', fontsize=fontsize_legend_axis)
            plt.xticks(fontsize=fontsize_legend_axis)
            plt.yticks(fontsize=fontsize_legend_axis)
            plt.grid()
            plt.show()


# main used for test output
if __name__ == "__main__":
    print("Neural network tests")

    X, Y = load_monk(1)

    nn = NeuralNetwork({'seed': 0,
                        'layers': [
                            {"neurons": len(one_hot(X[0])[0]), "activation": "linear"},
                            # input only for dimension, insert linear
                            {"neurons": 10, "activation": "tanh"},
                            {"neurons": 1, "activation": "tanh"}  # output
                        ],
                        'solver': 'sgd',
                        "problem": "classification"
                        })

    nn.fit(X=one_hot(X[0]),
           labels=[[i] for i in X[1]],
           X_validation=one_hot(Y[0]),
           labels_validation=[[i] for i in Y[1]],
           hyperparameters={"lambda": 0, "stepsize": 1, "momentum": 0.5, "epsilon": 0.009},
           epochs=1000,
           batch_size=32, )

    print("\nMean square error: train set")
    print(nn.score_mse(X=one_hot(X[0]), labels=[[i] for i in X[1]]))

    print("\nMean square error: test set")
    print(nn.score_mse(X=one_hot(Y[0]), labels=[[i] for i in Y[1]]))

    print("\nClassification accuracy training set:")
    print(nn.score_accuracy(one_hot(X[0]), [[i] for i in X[1]]))

    print("\nClassification accuracy test set:")
    print(nn.score_accuracy(one_hot(Y[0]), [[i] for i in Y[1]]))

    nn.plot_graph()