import numpy as np
from scipy.spatial import distance

def mean_squared_error(output: list, target: list):
    """
    Error function used to evaluate performance in monk dataset (classification)
    :param output: list [h(x_1), h(x_2), ...]. Outputs of the net for all sample
    :param target: list [y_1, y_2, ...]. Reals output of all samples
    :return: mse value
    """
    if len(output) != len(target):
        raise Exception("Dimension error")

    difference = np.subtract(output, target)
    squared = np.square(difference)
    mse = squared.mean()

    return mse


def classification_accuracy(output, target):
    """
    it measures the accuracy of the model
    :param output: the output of the model already "tresholded" respect to an input X
    :param target: the expected output for the input X
    :return: the classification accuracy of the model in percentage
    """

    if len(output) != len(target):
        raise Exception("Dimension error")

    correct = 0
    wrong = 0

    for i in range(0, len(target)):
        if output[i] == target[i]:
            correct += 1
        else:
            wrong += 1
    return correct / (correct + wrong)


def mean_euclidean_error(output: list, target: list):
    """
    Error function used to evaluate performance in cup of micheli (regression)
    :param output: list of list [[x_1, y_1, z_1, ...], [x_2, y_2, z_2, ...], ...] Outputs of the net for each point
    :param target: list of list [[x_1, y_1, z_1, ...], [x_2, y_2, z_2, ...], ...] Output real of points
    :return: mee value
    """
    if len(output) != len(target):
        raise Exception("Dimension error")

    difference = np.subtract(output, target)
    square = np.square(difference)
    squared = np.sqrt(np.sum(square, axis=1))
    mee = squared.mean()

    return mee



# main used for test output
if __name__ == "__main__":
    print("Score function tests")
    a = (1, 2, 3)
    b = (4, 5, 6)
    dst1 = distance.euclidean(a, b)
    dst2 = distance.euclidean(a, b)
    print((dst1+dst2)/2)

    a = [[1, 2, 3], [1, 2, 3]]
    b = [[4, 5, 6], [4, 5, 6]]
    dst = mean_euclidean_error(a, b)
    print(dst)