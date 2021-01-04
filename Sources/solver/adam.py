import numpy as np

from Sources.solver.iter_utility import __forward_pass, __backward_pass
from Sources.tools.score_function import mean_squared_error, classification_accuracy
from Sources.tools.useful import batch, unison_shuffle


def adam(X, labels, weights: dict, layers: dict, hyperparameters: dict, max_epochs: int, batch_size: int,
         shuffle: bool, X_validation, labels_validation):
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

    # needed to plot graph
    history = {}
    accuracy_train = []
    accuracy_validation = []
    mse_train = []
    mse_validation = []

    # momentum variables
    momentum_1_w = {}
    momentum_2_w = {}
    momentum_1_w_cap = {}
    momentum_2_w_cap = {}

    momentum_1_b = {}
    momentum_2_b = {}
    momentum_1_b_cap = {}
    momentum_2_b_cap = {}

    # hyperparam
    beta_1 = 0.9
    beta_2 = 0.999
    epsilon_adam = 1e-8

    # inizialization momentum variables
    for j in range(1, len(layers)):
        momentum_1_w["W" + str(j)] = np.zeros(weights["W" + str(j)].shape)
        momentum_2_w["W" + str(j)] = np.zeros(weights["W" + str(j)].shape)
        momentum_1_w_cap["W" + str(j)] = np.zeros(weights["W" + str(j)].shape)
        momentum_2_w_cap["W" + str(j)] = np.zeros(weights["W" + str(j)].shape)

        momentum_1_b["b" + str(j)] = np.zeros(weights["b" + str(j)].shape)
        momentum_2_b["b" + str(j)] = np.zeros(weights["b" + str(j)].shape)
        momentum_1_b_cap["b" + str(j)] = np.zeros(weights["b" + str(j)].shape)
        momentum_2_b_cap["b " + str(j)] = np.zeros(weights["b" + str(j)].shape)

    for i in range(0, max_epochs):

        for Xi, Yi in batch(X, labels, batch_size):  # get batch of x and y

            # forward propagatiom
            output, forward_cache = __forward_pass(Xi, weights, layers, True)

            # backward propagation
            deltaW, deltab = __backward_pass(output, np.array(Yi), weights, forward_cache, layers)

            # adjusting weigths
            for j in range(1, len(layers)):
                # update moment estimates
                momentum_1_w["W" + str(j)] = ((1 - beta_1) * deltaW["W" + str(j)]) + (
                            beta_1 * momentum_1_w["W" + str(j)])
                momentum_2_w["W" + str(j)] = ((1 - beta_2) * (deltaW["W" + str(j)] ** 2)) + (
                            beta_2 * momentum_2_w["W" + str(j)])
                momentum_1_b["b" + str(j)] = ((1 - beta_1) * deltab["b" + str(j)]) + (
                            beta_1 * momentum_1_b["b" + str(j)])
                momentum_2_b["b" + str(j)] = ((1 - beta_2) * (deltab["b" + str(j)] ** 2)) + (
                            beta_2 * momentum_2_b["b" + str(j)])

                # compute bias correction
                momentum_1_w_cap["W" + str(j)] = momentum_1_w["W" + str(j)] / (1 - (beta_1 ** (i + 1)))
                momentum_2_w_cap["W" + str(j)] = momentum_2_w["W" + str(j)] / (1 - (beta_2 ** (i + 1)))
                momentum_1_b_cap["b" + str(j)] = momentum_1_b["b" + str(j)] / (1 - (beta_1 ** (i + 1)))
                momentum_2_b_cap["b" + str(j)] = momentum_2_b["b" + str(j)] / (1 - (beta_2 ** (i + 1)))

                # update weight values
                weights["W" + str(j)] += ((hyperparameters["stepsize"] * momentum_1_w_cap["W" + str(j)]) /
                                          (np.sqrt(momentum_2_w_cap["W" + str(j)]) + epsilon_adam)) - \
                                         hyperparameters["lambda"] * weights["W" + str(j)]

                # update bias
                weights["b" + str(j)] += ((hyperparameters["stepsize"] * momentum_1_b_cap["b" + str(j)]) /
                                          (np.sqrt(momentum_2_b_cap["b" + str(j)]) + epsilon_adam))  # - \
                # hyperparameters["lambda"] * weights["b" + str(j)]

        output = __forward_pass(X, weights, layers, False)
        mse_train.append(mean_squared_error(output, labels))
        # save mse on validation data
        output_validation = __forward_pass(X_validation, weights, layers, False)
        mse_validation.append(mean_squared_error(output_validation, labels_validation))
        # save accuracy on training data
        accuracy_train.append(classification_accuracy(output, labels))
        # save accuracy on validation data
        accuracy_validation.append(classification_accuracy(output_validation, labels_validation))

        if mse_train[i] <= hyperparameters["epsilon"]:
            print("\nStopping condition raggiunta:\nerrore = " + str(mse_train[i]))
            break

        if shuffle:
            X, labels = unison_shuffle(X, labels)

        print("\nEpoch number " + str(i) + "\n->Error:", mse_train[i])

    history["mse_train"] = mse_train
    history["mse_validation"] = mse_validation
    history["acc_train"] = accuracy_train
    history["acc_validation"] = accuracy_validation

    print(mse_train[i])

    return history
