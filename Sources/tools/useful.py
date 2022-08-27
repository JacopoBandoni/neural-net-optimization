from itertools import product

import numpy as np
import random

from Sources.tools.preprocessing import one_hot


def batch(X, Y, n=1):
    """
    Method to return batches over X and y vector
    :param X: Any vector
    :param Y: Any vector
    :param n: how many batches
    :return: tuple of X,Y in batch
    """
    l = len(X)
    for ndx in range(0, l, n):
        yield np.array(X[ndx:min(ndx + n, l)]), np.array(Y[ndx:min(ndx + n, l)])


def unison_shuffle(X, y):
    unified = list(zip(X, y))  # create pair of x and y
    random.shuffle(unified)
    X, y = zip(*unified)  # unzip pair mantaining order
    return X, y


def k_fold(X, labels, fold_number):
    """
    Split dataset in k mutually exclusive subset, than produce k dataset train and validation
    Checked and verified, returns (for monks)
    X_train = (k, 132, 17)
    Y_train = (k, 132, )
    X_validation = (k, 33, 17)
    Y_validation = (k, 33)
    :param X:
    :param labels:
    :param model:
    :param configuration:
    :param fold_number:
    :param epochs:
    :return:
    """
    X, labels = unison_shuffle(X, labels)

    # dividing dataset
    partition_len = int(len(X) / fold_number)
    rest_of_patterns = len(X) % fold_number
    X_partitioned = [X[i:i + partition_len] for i in range(0, len(X) - rest_of_patterns, partition_len)]
    Y_partitioned = [labels[i:i + partition_len] for i in range(0, len(labels) - rest_of_patterns, partition_len)]

    X_train = []
    X_validation = []
    Y_train = []
    Y_validation = []
    for fold in range(0, fold_number):
        # creating partition mutually exclusive
        x_subset = X_partitioned[:fold] + X_partitioned[fold + 1:]
        x_train = np.concatenate(x_subset)

        y_subset = Y_partitioned[:fold] + Y_partitioned[fold + 1:]
        y_train = np.concatenate(y_subset)

        x_validation = np.array(X_partitioned[fold])
        y_validation = np.array(Y_partitioned[fold])

        X_train.append(x_train)
        Y_train.append(y_train)
        X_validation.append(x_validation)
        Y_validation.append(y_validation)

    return X_train, Y_train, X_validation, Y_validation


def grid_search(hyperparameters:dict):
    """
    Returns list of dictionary with all possible configurations
    :param hyperparameters:
    :return:
    """
    configurations = [dict(zip(hyperparameters, v)) for v in product(*hyperparameters.values())]
    return configurations


def hold_out(X, labels, percentage):
    X, labels = unison_shuffle(X, labels)

    portion = int((percentage*len(X))/100)

    X_train = X[:-portion]
    Y_train = labels[:-portion]
    X_test = X[-portion:]
    Y_test = labels[-portion:]

    return X_train, Y_train, X_test, Y_test



# main used for test output
if __name__ == "__main__":
    print("Useful function tests")
