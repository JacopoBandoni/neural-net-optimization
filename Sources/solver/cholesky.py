import time

import numpy as np
from scipy.linalg import cho_factor
from scipy.linalg import cho_solve
from scipy.linalg import solve_triangular
from scipy import stats
from numpy import linalg as LA

from Sources.tools.activation_function import *
from Sources.tools.score_function import *


def cholesky(X, labels, model, regularization, weights: dict, layers: dict,
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

    tic = time.perf_counter()

    # the cicle it's to begin the implementation of ELM with multiple ramdom layer
    for i in range(1, len(layers)-1):
        if i != len(layers):
            H = (H @ weights['W' + str(i)]) + weights['b' + str(i)]

        if layers[i]["activation"] == "sigmoid":
            H = sigmoid(H)
        if layers[i]["activation"] == "tanh":
            H = tanh(H)

    # andrebbe aggiunto len(x) secondo il report
    A = H.T @ H + len(X) * regularization * np.identity(layers[-2]["neurons"], float)
    B = H.T @ T
    C = np.linalg.cholesky(A)

    W2p = solve_triangular(C, B, lower=True)
    W2 = solve_triangular(C.T, W2p, lower=False)

    toc = time.perf_counter()

    print("\n")
    print(f"Seconds elapsed: {toc - tic:0.4f}")

    weights["W" + str(len(layers) -1)] = W2

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

    print("Error: ", history["error_train"])

    print("Condition number of A:", LA.cond(A))
    u, s, vh = LA.svd(H)
    print("Valori singolari di H: \n -> max: ", np.max(s), "\n -> min:", np.min(s))
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
