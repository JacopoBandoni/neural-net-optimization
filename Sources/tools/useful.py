import numpy as np
import random


def batch(X, y, n=1):
    """
    Method to return batches over X and y vector
    :param iterable: Any vector
    :param n: how many batches
    :return: tuple of X,y in batch
    """
    l = len(X)
    for ndx in range(0, l, n):
        yield np.array(X[ndx:min(ndx + n, l)]), np.array(y[ndx:min(ndx + n, l)])


def unison_shuffle(X, y):
    unified = list(zip(X, y))  # create pair of x and y
    random.shuffle(unified)
    X, y = zip(*unified)  # unzip pair mantaining order
    return X, y


# main used for test output
if __name__ == "__main__":
    print("Useful function tests")
