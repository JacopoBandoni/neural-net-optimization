import numpy as np


def uniform_weights(shape):
    return np.random.uniform(-0.7, 0.7, shape)


def xavier_init(shape):
    """
    #:param shape: the shape of the weight to initialize
    #:param n_i: the number of unit in input of the nn
    #:param n_o: the number of unit in output of the nn
    :return: a weight matrix
    """
    threshold = (np.sqrt(6)) / (np.sqrt(shape[0] + shape[1]))
    return np.random.uniform(-threshold, threshold, shape)


if __name__ == "__main__":
    print(0.7 * np.random.uniform(-0.7, 0.7, (4, 9)))
