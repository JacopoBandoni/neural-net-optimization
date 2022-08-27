import numpy as np

from Sources.solver.iter_utility import __forward_pass, __backward_pass
from Sources.tools.activation_function import *
from Sources.tools.useful import batch, unison_shuffle


def sgd(X, labels, model, hyperparameters: dict, max_epochs: int, batch_size: int, shuffle: bool,
        X_validation, labels_validation):
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

    deltaW_old = {}
    deltab_old = {}
    # needed to plot graph
    history = {}
    error_train = []
    error_validation = []
    accuracy_train = []
    accuracy_validation = []

    for i in range(0, max_epochs):

        for Xi, Yi in batch(X, labels, batch_size):  # get batch of x and y

            # forward propagatiom
            output, forward_cache = __forward_pass(Xi, model.weights, model.layers, True)

            # backward propagation
            deltaW, deltab = __backward_pass(output, np.array(Yi), model.weights, forward_cache, model.layers)

            # adjusting weigths
            for j in range(1, len(model.layers)):
                model.weights["W" + str(j)] += (hyperparameters["stepsize"]/len(Xi)) * deltaW["W" + str(j)] - \
                                         2*hyperparameters["lambda"] * model.weights["W" + str(j)]

                model.weights["b" + str(j)] += (hyperparameters["stepsize"]/len(Xi)) * deltab["b" + str(j)]

                if i != 0:
                    model.weights["W" + str(j)] += hyperparameters["momentum"] * deltaW_old["W" + str(j)]
                    model.weights["b" + str(j)] += (hyperparameters["stepsize"] / len(Xi)) * deltab_old["b" + str(j)]

            deltaW_old = deltaW
            deltab_old = deltab

        # save mse or mee
        if model.problem == "classification":
            error_train.append(model.score_mse(X, labels))
            error_validation.append(model.score_mse(X_validation, labels_validation))
            accuracy_train.append(model.score_accuracy(X, labels))
            accuracy_validation.append(model.score_accuracy(X_validation, labels_validation))
        elif model.problem == "regression":
            error_train.append(model.score_mee(X, labels))
            error_validation.append(model.score_mee(X_validation, labels_validation))
        else:
            raise Exception("Wrong problem statemenet (regression or classification)")

        if error_validation[i] <= hyperparameters["epsilon"]:
            print("Stopping condition raggiunta, errore = " + str(error_validation[i]))
            break

        if shuffle:
            X, labels = unison_shuffle(X, labels)

        # print("\nEpoch number " + str(i) + "\n->Error:", mse_train[i])

    history["error_train"] = error_train
    history["error_validation"] = error_validation
    history["acc_train"] = accuracy_train
    history["acc_validation"] = accuracy_validation

    return history
