import numpy as np

def weights_to_matrix(weights:list)->list:
    """
    Take the weights of the network and combine them to obtain a matrix of weights
    :param weights:
    :return: matrix of weights
    """
    pass



# main used for test output
if __name__ == "__main__":
    print("Useful function tests")

    units = [3, 1]
    features = 3

    # emulate the weight initialization in the neural network class
    weights = []
    # add input layer: row are neurons (1 = neuron 1) and column the corresponding weight for the i neuron
    weights.append({"W1": np.random.rand(units[0], features), "b1": np.zeros(units[0])})

    # add hidden layers: row are neurons (1 = neuron 1) and column the corresponding weight for the i neuron
    for i in range(1, len(units) - 1):
        weights.append([{'W' + str(i + 1): [np.random.rand(units[i], units[i - 1])],
                              "b" + str(i + 1): np.zeros(units[i])}])

    # add output layer: each row is one output with corresponding weights
    weights.append([{'W'+str(len(units)): [np.random.rand(units[-1], units[len(units) - 2])],
                          "b"+str(len(units)): np.zeros(units[-1])}])

    for l in weights:
        print(l)
