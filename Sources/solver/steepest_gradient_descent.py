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

    for i in range(0, epochs):
        # TODO -> sistemare batch ritorna un botto di roba
        # for Xi, Yi in batch(X, labels, batch_size):  # get batch of x and y

        # forward propagatiom
        output, forward_cache = __forward_pass(X, weights, layers)

        error = mean_squared_error(output, labels)
        errors.append(error)

        # backward propagation
        deltaW = __backward_pass(output, np.array(labels).mean(axis=0, keepdims=True), weights, forward_cache, layers)

        for j in range(1, len(layers)):
            if j == 2:
                weights["W" + str(j)] += hyperparameters["stepsize"] * deltaW["W" + str(j)].T

        print("\n->Error:", error)

    # perché fare lo shuffle alla fine?
    if shuffle:
        X, labels = unison_shuffle(X, labels)


def __forward_pass(x, weights: dict, layers):
    """
    Apply a forward pass in the network
    :param X: input vector of training set
    :param weights: weight dictionary of our network
    :param layers: layers configuration of our network
    :return: tuple (output, cache). output is the prediction of our network, cahce is all weight in intermediate steps
    """

    output = np.array(x)

    forward_cache = {"output0": output.mean(axis=0, keepdims=True)}

    for i in range(1, len(layers)):

        output = output @ weights['W' + str(i)] + weights['b' + str(i)]

        forward_cache["net" + str(i)] = output.mean(axis=0, keepdims=True)

        if layers[i]["activation"] == "sigmoid":
            output = sigmoid(output)
        elif layers[i]["activation"] == "tanh":
            output = tanh(output)
        elif layers[i]["activation"] == "linear":
            pass
        else:
            raise Exception("Activation function not recognized")

        forward_cache["output" + str(i)] = output.mean(axis=0, keepdims=True)

    return output, forward_cache


def __backward_pass(output, labels, weights: dict, forward_cache: dict, layers):
    delta_i = 0.0
    delta_next = []
    delta_prev = []
    deltaW = {}

    for layer in reversed(range(1, len(layers))):  # start from last layer

        # compute deltaW for external layer
        if layer == len(layers) - 1:

            for i in range(0, layers[layer]["neurons"]):

                # compute (yi - oi)
                delta_i = labels[i] - forward_cache['output' + str(layer)][0][i]

                print("\nlabels = " + str(labels[i]))
                print("output = " + str(forward_cache['output' + str(layer)][0][i]))
                print("label - output = " + str(delta_i))

                # compute (yi - oi)f'(neti)
                delta_i = delta_i * apply_d_activation(layers[layer]["activation"],
                                                       forward_cache['net' + str(layer)][0][i])

                print("delta" + str(i) + " = " + str(delta_i))

                # add delta to the list of delta's last layer
                delta_next.insert(0, delta_i)

            deltaW["W" + str(layer)] = np.array(delta_next) @ forward_cache['output' + str(layer - 1)]

        # compute deltaW for hidden layer
        else:
            for i in range(0, layers[layer]["neurons"]):

                delta_i = 0

                for j in range(0, layers[layer + 1]["neurons"]):
                    # compute sum( w*delta )
                    delta_i += weights["W" + str(layer + 1)][i, :] @ np.array(delta_next).T

                # compute sum( w*delta ) * f'(net)
                delta_i = delta_i * apply_d_activation(layers[layer]["activation"],
                                                       forward_cache['net' + str(layer)][0][i])
                # add delta to the list of delta_prev
                delta_prev.insert(0, delta_i)

            deltaW["W" + str(layer)] = np.array(delta_prev) @ forward_cache['output' + str(layer - 1)]

            delta_next = delta_prev
            delta_prev = []

    return deltaW


if __name__ == "__main__":
    print("Steepest gradient descent test")

    weights = {}
    X = ([[0, 1, 0], [0, 0, 1], [1, 0, 0], [1, 1, 0], [1, 1, 1]], [0, 0, 0, 1, 1])

    weights['W1'] = 0.7 * np.random.uniform(-0.7, 0.7, (3, 3))
    weights['b1'] = np.zeros((3, 1))

    weights['W2'] = 0.7 * np.random.uniform(-0.7, 0.7, (3, 1))
    weights['b2'] = np.zeros((1, 1))

    print(np.array(X[0]).mean(axis=0, keepdims=1))

    print(d_sigmoid(0.6))

    print( 0.23710834427459193 == 0.23710834427459193)
