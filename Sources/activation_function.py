import numpy as np


def sigmoid(X):
    result = 1 / (1 + np.exp(-X))
    return result


def tanh():
    pass


if __name__=="__main__":
    print("hello world")