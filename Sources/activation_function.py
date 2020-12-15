import numpy as np
import matplotlib.pyplot as plt

def sigmoid(X):
    result = 1 / (1 + np.exp(-X))
    return result


def tanh():
    pass


if __name__=="__main__":
    print("Activation function tests")
