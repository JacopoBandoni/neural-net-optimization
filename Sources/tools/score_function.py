import numpy as np


def mean_squared_error(output: list, target: list):
    """
    Error function used to evaluate performance in monk dataset (classification)
    :param output: list [h(x_1), h(x_2), ...]. Outputs of the net
    :param target: list [y_1, y_2, ...]. Reals output of samples
    :return: mse value
    """
    if len(output) != len(target):
        raise Exception("Dimension error")

    difference = np.subtract(output, target)
    squared = np.square(difference)
    mse = squared.mean()

    return mse


def mean_euclidean_error(output: list, target: list):
    """
    Error function used to evaluate performance in cup of micheli (regression)
    :param output: list of list [[x_1, y_1, z_1, ...], [x_2, y_2, z_2, ...], ...] Outputs of the net for each point
    :param target: list of list [[x_1, y_1, z_1, ...], [x_2, y_2, z_2, ...], ...] Output real of points
    :return: mee value
    """
    if len(output) != len(target):
        raise Exception("Dimension error")

    output = np.array(output)
    target = np.array(target)

    total_norm = []
    for i in range(0, len(output)):
        total_norm.append(np.linalg.norm(output[i] - target[i]))

    total_norm = np.array(total_norm)
    mee = total_norm.mean()

    return mee

# TODO complete the loss function
def mean_squared_loss(output: list, target: list, weights:list, lam:float):
    """
    Loss function used to evaluate performance in monk dataset (classification)
    :param output: list of list [[x_1, y_1, z_1, ...], [x_2, y_2, z_2, ...], ...] Outputs of the net for each point
    :param target: list of list [[x_1, y_1, z_1, ...], [x_2, y_2, z_2, ...], ...] Output real of points
    :param weights: all the weights of the network
    :param lam: hyperparameter
    :return:
    """
    if len(output) != len(target):
        raise Exception("Dimension error")

    difference = np.subtract(output, target)
    squared = np.square(difference)
    mse = squared.mean()


    # we have to transform our rappresentation matrix to a matrix of weights to compute norm

    penalty = np.linalg.norm()


    return mse


# main used for test output
if __name__ == "__main__":
    print("Score function tests")
