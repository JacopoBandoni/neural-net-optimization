import numpy as np

from Sources.solver.iter_utility import __forward_pass, __backward_pass, __backward_pass_extreme
from Sources.tools.score_function import mean_squared_error
from Sources.tools.useful import batch, unison_shuffle


def extreme_adam(X, labels, weights: dict, layers: dict, hyperparameters: dict, max_epochs: int, batch_size: int,
                 shuffle: bool):
    """
    Compute Adam just on the last layer with linear activ fun (least mean square problem)
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

    beta_1 = 0.9
    beta_2 = 0.999
    epsilon_adam = 1e-8

    num_batch = 0

    # inizialization momentum variables
    momentum_1_w = np.zeros(weights["W" + str(len(layers) - 1)].shape)
    momentum_2_w = np.zeros(weights["W" + str(len(layers) - 1)].shape)
    momentum_1_w_cap = np.zeros(weights["W" + str(len(layers) - 1)].shape)
    momentum_2_w_cap = np.zeros(weights["W" + str(len(layers) - 1)].shape)

    momentum_1_b = np.zeros(weights["b" + str(len(layers) - 1)].shape)
    momentum_2_b = np.zeros(weights["b" + str(len(layers) - 1)].shape)
    momentum_1_b_cap = np.zeros(weights["b" + str(len(layers) - 1)].shape)
    momentum_2_b_cap = np.zeros(weights["b" + str(len(layers) - 1)].shape)

    for i in range(0, max_epochs):

        for Xi, Yi in batch(X, labels, batch_size):  # get batch of x and y

            # forward propagatiom
            output, forward_cache = __forward_pass(Xi, weights, layers, True)

            # backward propagation
            deltaW, deltab = __backward_pass_extreme(output, np.array(Yi), forward_cache, layers)

            #print("calcolo deltaW dalla forward =\n" + str(momentum_1_w_cap))

            # update moment estimates
            momentum_1_w = ((1 - beta_1) * deltaW) + (beta_1 * momentum_1_w)
            momentum_2_w = ((1 - beta_2) * (deltaW ** 2)) + (beta_2 * momentum_2_w)
            momentum_1_b = ((1 - beta_1) * deltab) + (beta_1 * momentum_1_b)
            momentum_2_b = ((1 - beta_2) * (deltab ** 2)) + (beta_2 * momentum_2_b)

            #print("momentum 1 w =\n" + str(momentum_1_w_cap))
            #print("momentum 2 w =\n" + str(momentum_2_w_cap))

            # compute bias correction
            momentum_1_w_cap = momentum_1_w / (1 - (beta_1 ** (i + 1)))
            momentum_2_w_cap = momentum_2_w / (1 - (beta_2 ** (i + 1)))
            momentum_1_b_cap = momentum_1_b / (1 - (beta_1 ** (i + 1)))
            momentum_2_b_cap = momentum_2_b / (1 - (beta_2 ** (i + 1)))

            """
            print("momentum 1 w cap =\n" + str(momentum_1_w_cap))
            print("momentum 2 w cap =\n" + str(momentum_2_w_cap))
            print("formula adam senza reg =\n" + str(((hyperparameters["stepsize"] * momentum_1_w_cap) /
                                                      (np.sqrt(momentum_2_w_cap) + epsilon_adam))))
            """

            # update weight values
            weights["W" + str(len(layers) - 1)] += ((hyperparameters["stepsize"] * momentum_1_w_cap) /
                                                    (np.sqrt(momentum_2_w_cap) + epsilon_adam)) - \
                                                   hyperparameters["lambda"] * weights["W" + str(len(layers) - 1)]

            # print("pesi aggiornati =\n" + str(weights["W" + str(len(layers) - 1)]))

            # update bias
            weights["b" + str(len(layers) - 1)] += ((hyperparameters["stepsize"] * momentum_1_b_cap) /
                                                    (np.sqrt(momentum_2_b_cap) + epsilon_adam)) - \
                                                   hyperparameters["lambda"] * weights["b" + str(len(layers) - 1)]

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

    print(1 - (0.9 ** 2))
    print(np.sqrt(Y))