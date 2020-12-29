import numpy as np
from scipy.linalg import cho_factor
from scipy.linalg import cho_solve
from scipy.linalg import solve_triangular
from scipy import stats

from Sources.tools.activation_function import *
from Sources.tools.score_function import *


def cholesky(X, labels, regularization, weights: dict, layers: dict):
    """
    :param regularization: il termine di regolarizzazione
    :param X: Our whole training data
    :param labels: Our real output of training data
    :param weights: parameters alias weights of the network
    :param layers: layers information of our network (tho retrieve activation function of each layer)
    """

    T = labels
    H = np.array(X)

    # the cicle it's to begin the implementation of ELM with multiple ramdom layer
    for i in range(1, len(layers)-1):
        if i != len(layers):
            H = (H @ weights['W' + str(i)]) + weights['b' + str(i)]

        if layers[i]["activation"] == "sigmoid":
            H = sigmoid(H)
        if layers[i]["activation"] == "tanh":
            H = tanh(H)

    # andrebbe aggiunto len(x) secondo il report
    A = H.T @ H + np.identity(layers[-2]["neurons"], float)*regularization
    B = H.T @ T
    C = np.linalg.cholesky(A)

    W2p = solve_triangular(C, B, lower=True)
    W2 = solve_triangular(C.T, W2p, lower=False)

    weights["W" + str(len(layers) -1)] = W2

    """"
    TO VISUALIZE STABILITY ISSUE
    print("Stampo ( C(C.T) )W2 - (H.T)T")
    print(((C @ C.T) @ W2) - B)
    print()

    print("Stampo (C(C.T)) - A")
    print(((C @ C.T) - A))
    print()

    print("Stampo ( (H.T)H )W2 - (H.T)T")
    print(((H.T @ H) @ W2) - B)
    print()

    print("Stampo HW2 - T")
    print((H @ W2) - T)
    print()
    """


if __name__ == "__main__":
    print("Extreme learning tests: cholesky")
