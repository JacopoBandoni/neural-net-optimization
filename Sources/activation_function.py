import numpy as np
import matplotlib.pyplot as plt


def sigmoid(x):
    result = 1 / (1 + np.exp(-x))
    return result


def tanh(x):
    result = (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
    return result


if __name__ == "__main__":
    print("Activation function tests")
    print(sigmoid(-0.6))
