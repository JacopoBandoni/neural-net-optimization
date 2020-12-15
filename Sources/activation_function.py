import numpy as np
import matplotlib.pyplot as plt

def sigmoid(X):
    result = 1 / (1 + np.exp(-X))
    return result


def tanh():
    pass


if __name__=="__main__":
    input = np.linspace(-10, 10, 100)

    output = sigmoid(input)

    plt.plot(input, output)
    plt.xlabel("Input")
    plt.ylabel("Output")
    plt.title("Sigmoid")
    plt.show()