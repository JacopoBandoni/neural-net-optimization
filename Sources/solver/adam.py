import numpy as np

from Sources.solver.iter_utility import __forward_pass, __backward_pass
from Sources.tools.score_function import mean_squared_error
from Sources.tools.useful import batch, unison_shuffle


# TODO sempre da finire!

def adam(X, labels, weights: dict, layers: dict, hyperparameters: dict, max_epochs: int, batch_size: int,
         shuffle: bool):
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
    momentum_1 = {}
    momentum_2 = {}
    momentum_1_cap = {}
    momentum_2_cap = {}
    beta_1 = 0.9
    beta_2 = 0.999
    epsilon_adam = 1e-8

    num_batch = 0

    # inizialization momentum variables
    for j in range(1, len(layers)):
        momentum_1["W" + str(j)] = np.zeros(weights["W" + str(j)].shape)
        momentum_2["W" + str(j)] = np.zeros(weights["W" + str(j)].shape)
        momentum_1_cap["W" + str(j)] = np.zeros(weights["W" + str(j)].shape)
        momentum_2_cap["W" + str(j)] = np.zeros(weights["W" + str(j)].shape)

    for i in range(0, max_epochs):

        for Xi, Yi in batch(X, labels, batch_size):  # get batch of x and y

            # forward propagatiom
            output, forward_cache = __forward_pass(Xi, weights, layers, True)

            # backward propagation
            deltaW = __backward_pass(output, np.array(Yi), weights, forward_cache, layers)

            # adjusting weigths
            for j in range(1, len(layers)):

                # update moment estimates
                momentum_1["W" + str(j)] = ((1 - beta_1) * deltaW["W" + str(j)]) + (beta_1 * momentum_1["W" + str(j)])
                momentum_2["W" + str(j)] = ((1 - beta_2) * (deltaW["W" + str(j)] ** 2)) + (beta_2 * momentum_2["W" + str(j)])

                # compute bias correction
                momentum_1_cap["W" + str(j)] = momentum_1["W" + str(j)] / (1 - (beta_1 ** (i+1)))
                momentum_2_cap["W" + str(j)] = momentum_2["W" + str(j)] / (1 - (beta_2 ** (i+1)))

                """
                print("EPOCH " + str(i) + "   BATCH " + str(num_batch) + "  LAYER " + str(j))
                print("deltaW:\n" + str(deltaW["W" + str(j)]))
                print("momentum 1:\n" + str(momentum_1["W" + str(j)]))
                print("momentum 2:\n" + str(momentum_2["W" + str(j)]))
                print("momentum cap 1:\n" + str(momentum_1_cap["W" + str(j)]))
                print("momentum cap 2:\n" + str(momentum_2_cap["W" + str(j)]))

                print("sqrt momentum 2:\n" + str((np.sqrt(momentum_2_cap["W" + str(j)]))))
                """

                # update weight values
                weights["W" + str(j)] += ((hyperparameters["stepsize"] * momentum_1_cap["W" + str(j)]) /
                                          (np.sqrt(momentum_2_cap["W" + str(j)]) + epsilon_adam)) - \
                                         hyperparameters["lambda"] * weights["W" + str(j)]

                # print("new Weight:\n" + str(weights["W" + str(j)]))

            num_batch += 1

        num_batch = 0
        output = __forward_pass(X, weights, layers, False)
        error = mean_squared_error(output, labels)
        errors.append(error)

        if error <= hyperparameters["epsilon"]:
            print("\nStopping condition raggiunta:\nerrore = " + str(error))
            break

        if shuffle:
            X, labels = unison_shuffle(X, labels)

        print("\nEpoch number " + str(i) + "\n->Error:", error)

# main used for test output
if __name__ == "__main__":
    print("Adam function tests")

    Y = np.array([[1e-8, 4e-9], [1e-10, 9e-7]])

    print(1- (0.9**2))
    print(np.sqrt(Y))
