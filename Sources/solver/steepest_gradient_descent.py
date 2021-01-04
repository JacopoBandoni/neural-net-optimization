import numpy as np

from Sources.solver.iter_utility import __forward_pass, __backward_pass
from Sources.tools.activation_function import *
from Sources.tools.score_function import mean_squared_loss, mean_squared_error
from Sources.tools.useful import batch, unison_shuffle


def sgd(X, labels, weights: dict, layers: dict, hyperparameters: dict, max_epochs: int, batch_size: int, shuffle: bool):
    """
    Compute steepest gradient descent, either batch or stochastic
    :param X: Our whole training data
    :param labels: Our real output of training data
    :param weights: parameters alias weights of the network
    :param layers: layers information of our network (tho retrieve activation function of each layer)
    :param hyperparameters: Parameters to tune our sgd
        learning_rate = [0, 1] alpha of our update step
        epsilon = [0,1] precision for the stopping criteria of algorithm
        lambda = [0, 1] lambda value for penalty term used to regularize model
    :param max_epochs: Number of epochs
    :param batch_size: Number of samples to compute after update parameters
    :param shuffle: Either shuffle or not shuffle our data
    :return:
    """

    errors = []
    deltaW_old = {}
    deltab_old = {}

    for i in range(0, max_epochs):

        for Xi, Yi in batch(X, labels, batch_size):  # get batch of x and y

            # forward propagatiom
            output, forward_cache = __forward_pass(Xi, weights, layers, True)

            # backward propagation
            deltaW, deltab = __backward_pass(output, np.array(Yi), weights, forward_cache, layers)

            # adjusting weigths
            for j in range(1, len(layers)):
                weights["W" + str(j)] += (hyperparameters["stepsize"]/len(Xi)) * deltaW["W" + str(j)] - \
                                         2*hyperparameters["lambda"] * weights["W" + str(j)]

                weights["b" + str(j)] += (hyperparameters["stepsize"]/len(Xi)) * deltab["b" + str(j)]

                if i != 0:
                    weights["W" + str(j)] += hyperparameters["momentum"] * deltaW_old["W" + str(j)]
                    weights["b" + str(j)] += (hyperparameters["stepsize"] / len(Xi)) * deltab_old["b" + str(j)]

            deltaW_old = deltaW
            deltab_old = deltab

        output = __forward_pass(X, weights, layers, False)
        error = mean_squared_error(output, labels)
        errors.append(error)

        if error <= hyperparameters["epsilon"]:
            print("\nStopping condition raggiunta:\nerrore = " + str(error))
            break

        if shuffle:
            X, labels = unison_shuffle(X, labels)

        print("\nEpoch number " + str(i) + "\n->Error:", error)


if __name__ == "__main__":
    print("Steepest gradient descent test")

    Y = np.array([[1, 2, 1, 3], [1, 1, 1, 1]])

    print(Y.mean(axis=0))
