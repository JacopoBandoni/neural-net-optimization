import numpy as np

from Sources.tools.activation_function import sigmoid, tanh
from Sources.tools.score_function import mean_squared_loss, mean_squared_error
from Sources.tools.useful import batch, unison_shuffle


def sgd(X, labels, weights: dict, layers: dict, problem:str, hyperparameters:dict, epochs: int, batch_size: int,
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

    for e in range(0, epochs):
        print("Epoch:", e)

        for x, y in batch(X, labels, batch_size):  # get batch of x and y
            print("Creating batch x:", x, ", Batch y:", y)

            outputs = []
            for sample in range(len(x)): # for each sample in batch
                output, forward_cache = __forward_pass(x[sample], weights, layers)

                # round results based on problem choose
                if problem == "classification": # if it's classification round based on output
                    if layers[-1]["activation"] == "sigmoid":   # 0,1
                        if output>= 0.5:
                            output = 1
                        else:
                            output = 0

                    elif layers[-1]["activation"] == "tanh":    # -1, 1
                        if output >= 0:
                            output = 1
                        else:
                            output = 0
                outputs.append(output)

            error = mean_squared_error(outputs, y)

            #loss = mean_squared_loss(outputs, y, weights, hyperparameters["lambda"], len(layers))

            print("->Error:", error, ", Loss", 0)

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
    forward_cache = {}  # here we will store status in forward pass
    layer_input = np.array([x]).transpose()  # we don't update original values, the transpose is to match dimension

    for layer in range(1, len(layers)):
        W = weights['W' + str(layer)]  # retrieve corresponding weights

        b = np.array(weights['b' + str(layer)])  # retrieve corresponding bias and do transpose

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
