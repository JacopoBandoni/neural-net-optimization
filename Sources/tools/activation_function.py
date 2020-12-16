import numpy as np
import matplotlib.pyplot as plt


def sigmoid(x):
    result = 1 / (1 + np.exp(-x))
    return result


def d_sigmoid(x):
    result = sigmoid(x) * (1 - sigmoid(x))
    return result


def tanh(x):
    result = (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
    return result


def d_tanh(x):
    result = 1-(tanh(x)**2)
    return result

if __name__ == "__main__":
    print("Activation function tests")
    print(d_tanh(-0.6))
