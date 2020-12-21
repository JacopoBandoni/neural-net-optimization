import numpy as np


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
def mean_squared_loss(output: list, target: list, weights: dict, lam: float, layers_number: int):
    """
    Error function used to evaluate performance in monk dataset (classification)
    :param output: list [h(x_1), h(x_2), ...]. Outputs of the net for all sample
    :param target: list [y_1, y_2, ...]. Real output of all samples
    :param weights: dictionary of weight and bias
    :param lam: hyperparameter used to compute penalty term
    :param layers_number: number of layer used to retrieve weights
    :return: loss score
    """
    if len(output) != len(target):
        raise Exception("Dimension error")

    mse = mean_squared_error(output, target)

    # we take vector of all weights in the network
    all_weights = []
    for layer in range(1, layers_number):
        W = np.array(weights['W ' + str(layer)])  # retrieve corresponding weights of layer
        all_weights.append(W.flatten())  # move to one dimensional array and add to the list

    all_weights = np.array(all_weights).flatten()  # move to one dimensional array with all weights

    penalty_term = lam * (np.linalg.norm(all_weights))

    mse_loss = mse + penalty_term

    return mse_loss


# main used for test output
if __name__ == "__main__":
    print("Score function tests")
