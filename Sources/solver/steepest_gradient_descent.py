import numpy as np

from Sources.tools.activation_function import sigmoid, tanh
from Sources.tools.score_function import mean_squared_loss, mean_squared_error
from Sources.tools.useful import batch, unison_shuffle


def sgd(X, labels, weights: dict, layers: dict, learning_rate: float, epsilon: float, epochs: int, batch_size: int,
        shuffle: bool):
    """
    :param X: training data where shapes match with initialization values
    :param labels: training y data where shapes match with initialization values
    :param learning_rate: alpha value to update weights
    :return:
    """

    errors = []

    for e in range(0, epochs):
        print("Epoch:", e)
        for x, y in batch(X, labels, batch_size):  # get batch of x and y
            print("Batch x:", x, ", Batch y:", y)


        #TODO here insert code
        __forward_pass(X, weights, layers)

        if shuffle:
            X, labels = unison_shuffle(X, labels)
        """

        sample_batch = []

        output, forward_cache = forward_pass(X, weights, layers)

        error = mean_squared_error(output, labels)
        
        """


def __forward_pass(X, weights: dict, layers):
    """
    Apply a forward pass in the network
    :param X: input vector of training set
    :param weights: weight dictionary of our network
    :param layers: layers configuration of our network
    :return: tuple (output, cache). output is the prediction of our network, cahce is all weight in intermediate steps
    """
    forward_cache = {}  # here we will store status in forward pass
    layer_input = X  # we don't update original values

    for layer in range(1, len(layers)):
        W = weights['W ' + str(layer)]  # retrieve corresponding weights
        b = weights['b ' + str(layer)]  # retrieve corresponding bias
        activation = layers[layer]["activation"]  # retrieve corresponding activation function

        # compute first linear activation of one layer
        Z = np.dot(W, layer_input) + b  # multiply weight dot input adding bias
        # compute the non-linear activation through that layer
        if activation == "sigmoid":
            layer_output = sigmoid(Z)
        elif activation == "tanh":
            layer_output = tanh(Z)
        else:
            raise Exception("Activation function not recognized")

        # store status
        forward_cache['Z' + str(layer)] = Z
        forward_cache['layer_output' + str(layer - 1)] = layer_input

        layer_input = layer_output  # update input to feed the next layer

    return layer_output, forward_cache


def __compute_loss(output, labels, weights: dict, lam: float, layers_number: int):
    return mean_squared_loss(output, labels, weights, lam, layers_number)


def __backward_pass(output, labels, weights: dict, forward_cache: dict, layers):
    grads = {}
    number_of_layer = len(layers)

    labels = np.array(labels)
    y = labels.reshape(output.shape)  # labels are not same shape as output to match dimensions

    for layer in reversed(range(1, number_of_layer)):  # start from last layer
        W = weights['W ' + str(layer)]  # retrieve corresponding weights
        b = weights['b ' + str(layer)]  # retrieve corresponding bias
        activation = layers[layer]["activation"]  # retrieve corresponding activation function

        output_previous = forward_cache['layer_output' + str(layer - 1)]

        activation_backward()


def __activation_backward(activation):
    if activation == "simoid":
        Z_derivative = 0
    elif activation == "tanh":
        pass
    else:
        raise Exception("Activation function not recognized")


if __name__ == "__main__":
    print("Steepest gradient descent test")
