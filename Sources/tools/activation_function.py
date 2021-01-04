import numpy as np


def sigmoid(x):
    x = np.array(x)
    result = 1 / (1 + np.exp(-x))
    return result


def d_sigmoid(x):
    x = np.array(x)
    result = sigmoid(x) * (1 - sigmoid(x))
    return result


def tanh(x):
    x = np.array(x)
    result = (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
    return result


def d_tanh(x):
    x = np.array(x)
    result = 1 - (tanh(x) ** 2)
    return result


def apply_activation(activation, v):
    if activation == "sigmoid":
        return sigmoid(v)
    elif activation == "tanh":
        return tanh(v)
    elif activation == "linear":
        return v
    else:
        raise Exception("Activation function not recognized")


def apply_d_activation(activation, v):
    if activation == "sigmoid":
        return d_sigmoid(v)
    elif activation == "tanh":
        return d_tanh(v)
    elif activation == "linear":
        return np.ones(v.shape)
    else:
        raise Exception("Activation function not recognized")


if __name__ == "__main__":
    print("Activation function tests")
    print(tanh([1, 2, 3, 4]))
