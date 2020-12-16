import numpy as np

from Sources.tools.activation_function import sigmoid, tanh


def sgd(X, labels, weights, learning_rate: float, epsilon: float):
    """
    :param X: training data where shapes match with initialization values
    :param labels: training y data where shapes match with initialization values
    :param learning_rate: alpha value to update weights
    :return:
    """


def forward_pass(X, weights, layers):
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

        Z, layer_output = activation_forward(layer_input, W, b, activation)

        # store status
        forward_cache['Z' + str(layer)] = Z
        forward_cache['layer_output' + str(layer - 1)] = layer_input

        layer_input = layer_output  # update input to feed the next layer

    return layer_output, forward_cache


def activation_forward(layer_input, W, b, activation):
    """
    Apply The computation of single layer, The matrix of neurons weights is multiplyed to input, adding bias
    :param layer_input: Input values
    :param W: weight matrix (each neuron has a vector)
    :param b: bias vector (each neuron has scalar)
    :param activation: string of activation to perform
    :return: tuple (Z, layer_output). Z is weights * input + bias, layer_output is activation applied to Z
    """
    Z = np.dot(W, layer_input) + b  # multiply weight dot input adding bias

    if activation == "sigmoid":
        layer_output = sigmoid(Z)

    elif activation == "tanh":
        layer_output = tanh(Z)
    else:
        raise Exception("Activation function not recognized")

    return Z, layer_output


def compute_loss(output, labels):
    pass