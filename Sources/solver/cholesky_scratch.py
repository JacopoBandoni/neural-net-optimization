import numpy as np
import math
from scipy import stats

from Sources.tools.activation_function import *
from Sources.tools.score_function import *

def __solve_upper_system(L, b):
    """
    Solve upper triangular system with backward substitution
    :param L:
    :param b:
    :return:
    """
    x = np.zeros_like(b, dtype=float)
    x[-1] = (b[-1][0] / L[-1][-1])
    for i in reversed(range(0, len(L)-1)):
        x[i] = (b[i] - L[i][i:] @ x[i:]) / float(L[i][i])
    return np.array(x)

def __solve_lower_system(L, b):
    """
    Solve lower triangular system with forward substitution
    :param L:
    :param b:
    :return:
    """
    x = np.zeros_like(b, dtype=float)
    x[0] = (b[0][0] / L[0][0])
    for i in range(1, len(L)):
        x[i] = (b[i] - L[i][:i] @ x[:i]) / float(L[i][i])

    return np.array(x)


def __cholesky_decomposition(A):
    """
    Performs a Cholesky decomposition of A
    :param A: must be a symmetric and positive definite matrix
    :return: the lower variant triangular matrix, L
    """
    n = len(A)

    # Create zero matrix for L
    L = [[0.0] * n for i in range(n)]

    # Perform the Cholesky decomposition
    for i in range(n):
        for k in range(i + 1):
            tmp_sum = sum(L[i][j] * L[k][j] for j in range(k))

            if (i == k):  # Diagonal elements
                # LaTeX: l_{kk} = \sqrt{ a_{kk} - \sum^{k-1}_{j=1} l^2_{kj}}
                L[i][k] = math.sqrt(A[i][i] - tmp_sum)
            else:
                # LaTeX: l_{ik} = \frac{1}{l_{kk}} \left( a_{ik} - \sum^{k-1}_{j=1} l_{ij} l_{kj} \right)
                L[i][k] = (1.0 / L[k][k] * (A[i][k] - tmp_sum))

    return np.array(L)


def cholesky_scratch(X, labels, model, regularization, weights: dict, layers: dict,
             X_validation, labels_validation):
    """
    :param regularization: il termine di regolarizzazione
    :param X: Our whole training data
    :param labels: Our real output of training data
    :param weights: parameters alias weights of the network
    :param layers: layers information of our network (tho retrieve activation function of each layer)
    """

    T = labels
    H = np.array(X)

    # the cicle it's to begin the implementation of ELM with multiple random layer
    for i in range(1, len(layers) - 1):
        if i != len(layers):
            H = (H @ weights['W' + str(i)]) + weights['b' + str(i)]

        if layers[i]["activation"] == "sigmoid":
            H = sigmoid(H)
        if layers[i]["activation"] == "tanh":
            H = tanh(H)

    # andrebbe aggiunto len(x) secondo il report
    A = H.T @ H + np.identity(layers[-2]["neurons"], float) * regularization
    B = H.T @ T

    C = __cholesky_decomposition(A)

    W2p = __solve_lower_system(C, B)
    W2 = __solve_upper_system(C.T, W2p)

    weights["W" + str(len(layers) - 1)] = W2

    # save mse or mee
    history = {}
    if model.problem == "classification":
        history["error_train"] = [model.score_mse(X, labels)]
        if X_validation is not None:
            history["error_validation"] = [model.score_mse(X_validation, labels_validation)]
        history["acc_train"] = [model.score_accuracy(X, labels)]
        if X_validation is not None:
            history["acc_validation"] = [model.score_accuracy(X_validation, labels_validation)]
    elif model.problem == "regression":
        history["error_train"] = [model.score_mee(X, labels)]
        if X_validation is not None:
            history["error_validation"] = [model.score_mee(X_validation, labels_validation)]
    else:
        raise Exception("Wrong problem statemenet (regression or classification)")


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

    return history


if __name__ == "__main__":
    print("Extreme learning tests: cholesky")

    A = [[30, -3, 50],
         [1, 40, 16],
         [30, -5, -50],
         [23, 40, 7]]
    A = np.array(A)

    b = [[4], [5], [-1]]

    L = __cholesky_decomposition(A.T @ A)
    print(L)
    B = __solve_lower_system(L, b)
    print(B)
    print(__solve_upper_system(np.array(L).transpose(), B))
