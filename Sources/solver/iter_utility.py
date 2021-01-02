import numpy as np

from Sources.tools.activation_function import apply_d_activation, apply_activation


def __forward_pass(x, weights: dict, layers, cache):
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

        if cache:
            forward_cache["net" + str(i)] = output

        output = apply_activation(layers[i]["activation"], output)

        if cache:
            forward_cache["output" + str(i)] = output

    if cache:
        return output, forward_cache
    else:
        return output


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
                # compute (yi - oi) for each pattern
                delta_i = labels[:, [i]] - forward_cache["output" + str(layer)][:, [i]]

                # compute delta_i = (yi - oi)f'(neti) for each pattern
                delta_i = delta_i.T * apply_d_activation(layers[layer]["activation"],
                                                         forward_cache['net' + str(layer)][:, [i]].T)

                delta_next.append(delta_i[0].tolist())

                # compute oj(yi - oi)f'(neti) for each pattern and do the mean on all the pattern
                deltaW["W" + str(layer)].append((np.array(forward_cache['output' + str(layer - 1)]) * delta_i.T)
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
                deltaW["W" + str(layer)].append((np.array(forward_cache['output' + str(layer - 1)]) * delta_i.T)
                                                .mean(axis=0).tolist())

            delta_next = delta_prev
            delta_prev = []
            deltaW["W" + str(layer)] = np.array(deltaW["W" + str(layer)]).T

    return deltaW
