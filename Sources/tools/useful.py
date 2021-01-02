import numpy as np
import random


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


# main used for test output
if __name__ == "__main__":
    print("Useful function tests")