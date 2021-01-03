import numpy as np


def uniform_weights(shape):
    return np.random.uniform(-0.7, 0.7, shape)


def xavier_init(shape, n_i, n_o):
    """
    #:param shape: the shape of the weight to initialize
    #:param n_i: the number of unit in input of the nn
    #:param n_o: the number of unit in output of the nn
    :return: a weight matrix
    """

    return np.random.rand(shape[0], shape[1]) * np.sqrt(1 / (n_i + n_o))


if __name__ == "__main__":
    print(0.7 * np.random.uniform(-0.7, 0.7, (4, 9)))
