import numpy as np

from Sources.tools.activation_function import *
from Sources.tools.score_function import mean_squared_loss, mean_squared_error, classification_accuracy
from Sources.tools.useful import batch, unison_shuffle


def sgd(X, labels, weights: dict, layers: dict, problem: str, hyperparameters: dict, epochs: int, batch_size: int,
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
    :param epochs: Number of epochs
    :param batch_size: Number of samples to compute after update parameters
    :param shuffle: Either shuffle or not shuffle our data
    :return:
    """

    errors = []
    deltaW_old = {}

    for i in range(0, epochs):
        # TODO -> sistemare batch ritorna un botto di roba
        # for Xi, Yi in batch(X, labels, batch_size):  # get batch of x and y

        # forward propagatiom
        output, forward_cache = __forward_pass(X, weights, layers)

        error = mean_squared_error(output, labels)
        errors.append(error)

        # backward propagation
        deltaW = __backward_pass(output, np.array(labels), weights, forward_cache, layers)

        # adjusting weigths
        for j in range(1, len(layers)):
            weights["W" + str(j)] += hyperparameters["stepsize"] * deltaW["W" + str(j)] - \
                                     hyperparameters["lambda"]*weights["W" + str(j)]\

            if i != 0:
                weights["W" + str(j)] += hyperparameters["momentum"]*deltaW_old["W" + str(j)]

        deltaW_old = deltaW

        print("\n->Error:", error)


def __forward_pass(x, weights: dict, layers):
    """
    Apply a forward pass in the network
    :param X: input vector of training set
    :param weights: weight dictionary of our network
    :param layers: layers configuration of our network
    :return: tuple (output, cache). output is the prediction of our network, cahce is all weight in intermediate steps
    """

    output = np.array(x)
    forward_cache = {"output0": output}

    for i in range(1, len(layers)):

        output = output @ weights['W' + str(i)] + weights['b' + str(i)]

        forward_cache["net" + str(i)] = output

        if layers[i]["activation"] == "sigmoid":
            output = sigmoid(output)
        elif layers[i]["activation"] == "tanh":
            output = tanh(output)
        elif layers[i]["activation"] == "linear":
            pass
        else:
            raise Exception("Activation function not recognized")

        forward_cache["output" + str(i)] = output

    return output, forward_cache


# TODO SBAGLIATO CAMBIARE, AGGIORNAMENTO SBAGLIATO!!!!
def __backward_pass(output, labels, weights: dict, forward_cache: dict, layers):
    delta_i = 0
    delta_prev = []
    delta_next = []
    deltaW = {}

    for layer in reversed(range(1, len(layers))):  # start from last layer

        deltaW["W" + str(layer)] = []

        # compute deltaW for external layer
        if layer == len(layers) - 1:

            # compute the vector of deltaW for each neuron in output
            for i in range(0, layers[layer]["neurons"]):

                # compute (yi - oi) for each pattern #TODO sottrazione tra vettori si può fare!
                delta_i = labels[:, i] - forward_cache["output" + str(layer)][:, i]

                # compute delta_i = (yi - oi)f'(neti) for each pattern
                # (TODO sostituibile come sotto come molt di matrici (*) element-wise)
                for j in range(0, len(output)):
                    delta_i[j] = delta_i[j] * apply_d_activation(layers[layer]["activation"],
                                                                 forward_cache['net' + str(layer)][j][i])

                delta_next.append(delta_i)

                # compute oj(yi - oi)f'(neti) for each pattern and do the mean on all the pattern
                deltaW["W" + str(layer)].append(
                    (np.array(forward_cache['output' + str(layer - 1)] * np.array([delta_i]).T))
                    .mean(axis=0).tolist())

            deltaW["W" + str(layer)] = np.array(deltaW["W" + str(layer)]).T

        # compute deltaW for hidden layer
        else:

            # compute the vector of deltaW for each neuron in the hidden layer
            for i in range(0, layers[layer]["neurons"]):

                # compute sum( w*delta ) for each pattern (delta next is made of all the pattern)
                delta_i = weights["W" + str(layer + 1)][[i], :] @ np.array(delta_next)

                # compute sum( w*delta ) * f'(net) for each pattern
                delta_i = delta_i * apply_d_activation(layers[layer]["activation"],
                                                       forward_cache['net' + str(layer)][:, [i]].T)

                delta_prev.append(delta_i[0].tolist())

                # compute o * sum( w*delta ) * f'(net) for each pattern and do the mean
                deltaW["W" + str(layer)].append((np.array(forward_cache['output' + str(layer - 1)] * delta_i.T))
                                                .mean(axis=0).tolist())

            delta_next = delta_prev
            delta_prev = []
            deltaW["W" + str(layer)] = np.array(deltaW["W" + str(layer)]).T

    return deltaW


if __name__ == "__main__":
    print("Steepest gradient descent test")

    weights = {}
    X = ([[0, 1, 0], [0, 0, 1], [1, 0, 0], [1, 1, 0], [1, 1, 1]], [0, 0, 0, 1, 1])

    weights['W1'] = 0.7 * np.random.uniform(-0.7, 0.7, (3, 3))
    weights['b1'] = np.zeros((3, 1))

    weights['W2'] = 0.7 * np.random.uniform(-0.7, 0.7, (3, 1))
    weights['b2'] = np.zeros((1, 1))

    Y = np.array([[1, 2, 1, 3]])
    Y2 = np.array([1, 1])

    print(Y)
    print(Y.shape)
    print(Y[0])
    print(Y[0].shape)
