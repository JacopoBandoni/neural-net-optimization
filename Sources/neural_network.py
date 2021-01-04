import numpy as np

from Sources.solver.adam import adam
from Sources.solver.cholesky import cholesky
from Sources.solver.extreme_adam import extreme_adam
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
                 settings["solver"] == "extreme_adam" or
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

    def fit(self, X, labels, X_validation, labels_validation,  hyperparameters: dict, epochs=1, batch_size=32, shuffle=True,
            ):
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
            print("\nRunning sgd")
            sgd(X, labels, self.weights, self.layers, hyperparameters, epochs, batch_size, shuffle,
                X_validation, labels_validation)

        elif self.solver == "adam":
            print("\nRunning adam")
            adam(X, labels, self.weights, self.layers, hyperparameters, epochs, batch_size, shuffle,
                 X_validation, labels_validation)

        elif self.solver == "extreme_adam":
            print("\nRunning adam")
            extreme_adam(X, labels, self.weights, self.layers, hyperparameters, epochs, batch_size, shuffle)

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

# main used for test output
if __name__ == "__main__":
    print("Neural network tests")

    """
    X = ([[0, 1, 0], [0, 0, 1], [1, 0, 0], [1, 1, 0], [1, 1, 1]], [[0, 0], [0, 0], [0, 0], [1, 0], [1, 1]])
    Y = ([[0, 0, 0], [0, 1, 0], [1, 1, 0], [0, 1, 1], [1, 1, 1]], [[0, 0], [0, 0], [1, 0], [0, 1], [1, 1]])

    nn = NeuralNetwork({'seed': 0,
                        'layers': [
                            {"neurons": len(X[0][0]), "activation": "linear"},
                            # input only for dimension, insert linear
                            {"neurons": 6, "activation": "sigmoid"},
                            {"neurons": 2, "activation": "sigmoid"}  # output
                        ],
                        'solver': 'sgd',
                        "problem": "classification"
                        })

    nn.fit(X=X[0], labels=X[1], hyperparameters={"lambda": 0, "stepsize": 1, "momentum": 0.9, "epsilon": 0.001}, epochs=1000)

    # print("\nPrediction")
    # print(nn.predict(one_hot(X[0])).mean(axis=0))
    # print(np.array([[i] for i in X[1]]).mean(axis=0))

    print("\nMean square error: train set")
    print(nn.score(X=X[0], labels=X[1]))

    print("\nMean square error: test set")
    print(nn.score(X=Y[0], labels=Y[1]))

    """

    X, Y = load_monk(3)

    nn = NeuralNetwork({'seed': 0,
                        'layers': [
                            {"neurons": len(one_hot(X[0])[0]), "activation": "linear"},
                            # input only for dimension, insert linear
                            {"neurons": 100, "activation": "tanh"},
                            {"neurons": 1, "activation": "linear"}  # output
                        ],
                        'solver': 'extreme_adam',
                        "problem": "classification"
                        })

    nn.fit(X=one_hot(X[0]),
           labels=[[i] for i in X[1]],
           hyperparameters={"lambda": 0, "stepsize": 0.001, "momentum": 0.9, "epsilon": 0.009},
           epochs=1000,
           batch_size=32,)

    print("\nMean square error: train set")
    print(nn.score(X=one_hot(X[0]), labels=[[i] for i in X[1]]))

    print("\nMean square error: test set")
    print(nn.score(X=one_hot(Y[0]), labels=[[i] for i in Y[1]]))

    treshold_list_train = [1 if i > 0.5 else 0 for i in nn.predict(one_hot(X[0]))]

    treshold_list_test = [1 if i > 0.5 else 0 for i in nn.predict(one_hot(Y[0]))]

    print("\nClassification accuracy training set:")
    print(classification_accuracy(output=treshold_list_train, target=X[1]))

    print("\nClassification accuracy test set:")
    print(classification_accuracy(output=treshold_list_test, target=Y[1]))

