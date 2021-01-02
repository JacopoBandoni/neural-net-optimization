import numpy as np

from Sources.solver.iter_utility import __forward_pass, __backward_pass
from Sources.tools.score_function import mean_squared_error
from Sources.tools.useful import batch

# TODO sempre da finire!

def adam(X, labels, weights: dict, layers: dict, hyperparameters: dict, max_epochs: int, batch_size: int):
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
    momentum_1 = 0
    momentum_2 = 0
    beta_1 = 0.9
    beta_2 = 0.999
    epsilon_adam = 10 ** -8

    for i in range(0, max_epochs):

        for Xi, Yi in batch(X, labels, batch_size):  # get batch of x and y

            # forward propagatiom
            output, forward_cache = __forward_pass(Xi, weights, layers, True)

            # backward propagation
            deltaW = __backward_pass(output, np.array(Yi), weights, forward_cache, layers)

            # adjusting weigths
            for j in range(1, len(layers)):
                momentum_1 = (1 - beta_1) * deltaW + beta_1 * momentum_1
                momentum_2 = (1 - beta_2) * (deltaW ** 2) + (beta_2 * momentum_2)

                momentum_1 = momentum_1/(1-(beta_1**i))

                weights["W" + str(j)] += hyperparameters["stepsize"] * deltaW["W" + str(j)] - \
                                         hyperparameters["lambda"] * weights["W" + str(j)]
                if i != 0:
                    weights["W" + str(j)] += hyperparameters["momentum"] * deltaW_old["W" + str(j)]

            deltaW_old = deltaW

        output = __forward_pass(X, weights, layers, False)
        error = mean_squared_error(output, labels)
        errors.append(error)

        if error <= hyperparameters["epsilon"]:
            print("\nStopping condition raggiunta:\nerrore = " + str(error))
            break

        print("\nEpoch number " + str(i) + "\n->Error:", error)
