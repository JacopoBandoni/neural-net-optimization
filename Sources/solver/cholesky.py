import numpy as np
from scipy.linalg import cho_factor
from scipy.linalg import cho_solve
from scipy.linalg import solve_triangular
from scipy import stats

from Sources.neural_network import NeuralNetwork
from Sources.tools.activation_function import *
from Sources.tools.load_dataset import *
from Sources.tools.useful import *
from Sources.tools.preprocessing import *
from Sources.tools.score_function import *


def cholesky(X, labels, regularization, weights: dict, layers: dict):
    """
    :param X: Our whole training data
    :param labels: Our real output of training data
    :param weights: parameters alias weights of the network
    :param layers: layers information of our network (tho retrieve activation function of each layer)
    """

    print("stampo i dati X = " + str(np.array(X).shape))
    print("stampo i dati W = " + str(np.array(weights["W1"]).T.shape))

    T = labels
    H = ((np.array(X) @ np.array(weights["W1"]).T) + np.array(weights["b1"]).T)
    H = sigmoid(H)

    A = H.T @ H + np.identity(layers[1]["neurons"], float)*regularization # lambda = 0.03 andrebbe aggiunto len(x) secondo il report
    B = H.T @ T

    C = np.linalg.cholesky(A)

    W2p = solve_triangular(C, B, lower=True)

    W2 = solve_triangular(C.T, W2p, lower=False)

    print(np.linalg.cond(A))

    weights["W2"] = W2.T

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


if __name__ == "__main__":
    print("Extreme learning tests: cholesky")

    #X = ([[1, 2], [1, 3], [2, 3], [5, 7]], [1, 0, 1, 1])

    # prova con minst dataset e con cup dataset

    X, Y = load_monk(2)

    nn = NeuralNetwork({'seed': 0,
                        'layers': [
                            {"neurons": len(one_hot(X[0])[0]), "activation": "linear"},
                            # input only for dimension, insert linear
                            {"neurons": 3, "activation": "sigmoid"},
                            {"neurons": 1, "activation": "linear"}  # output
                        ],
                        'solver': 'sgd',
                        "problem": "classification"
                        })

    cholesky(X=one_hot(X[0]), labels=[[i] for i in X[1]], regularization=10, weights=nn.weights, layers=nn.layers)

    scoreTrain = nn.score(X=one_hot(X[0]), labels=[[i] for i in X[1]])
    scoreTest = nn.score(X=one_hot(Y[0]), labels=[[i] for i in Y[1]])

    print()
    print("Mean square error: train set")
    print(scoreTrain)

    print()
    print("Mean square error: test set")
    print(scoreTest)

    treshold_list_train = []

    for i in nn.predict(one_hot(X[0])):
        if i > 0.5:
            treshold_list_train.insert(len(treshold_list_train), 1)
        if i <= 0.5:
            treshold_list_train.insert(len(treshold_list_train), 0)

    treshold_list_test = []

    for i in nn.predict(one_hot(Y[0])):
        if i > 0.5:
            treshold_list_test.insert(len(treshold_list_test), 1)
        if i <= 0.5:
            treshold_list_test.insert(len(treshold_list_test), 0)

    print()
    print("Classification accuracy training set:")
    print(classification_accuracy(output=treshold_list_train, target=X[1]))

    print()
    print("Classification accuracy test set:")
    print(classification_accuracy(output=treshold_list_test, target=Y[1]))