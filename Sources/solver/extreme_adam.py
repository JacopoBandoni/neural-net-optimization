import numpy as np
import time

from Sources.solver.iter_utility import __forward_pass, __backward_pass, __backward_pass_extreme
from Sources.tools.score_function import mean_squared_error
from Sources.tools.useful import batch, unison_shuffle
import matplotlib.pyplot as plt
from numpy import linalg as LA

def extreme_adam(X, labels, model, hyperparameters: dict, max_epochs: int, batch_size: int, shuffle: bool,
                 X_validation, labels_validation):
    """
    Compute Adam just on the last layer with linear activ fun (least mean square problem)
    :param X: Our whole training data
    :param labels: Our real output of training data
    :param weights: parameters alias weights of the network
    :param layers: layers information of our network (tho retrieve activation function of each layer)
    :param hyperparameters: Parameters to tune our sgd
        learning_rate = [0, 1] alpha of our update step
        epsilon = [0,1] precision for the stopping criteria of algorithm
        lambda = [0, 1] lambda value for penalty term used to regularize model
    :param max_epochs: Number of epochs
    :param batch_size: Number of samples to compute after update parameters
    :param shuffle: Either shuffle or not shuffle our data
    :return:
    """
    # needed to plot graph
    history = {}
    accuracy_train = []
    accuracy_validation = []
    error_train = []
    error_validation = []
    norm_of_gradients = []
    variances = []
    losses = []

    errors = []

    beta_1 = 0.9
    beta_2 = 0.999
    epsilon_adam = 1e-8

    num_batch = 0

    # inizialization momentum variables
    momentum_1_w = np.zeros(model.weights["W" + str(len(model.layers) - 1)].shape)
    momentum_2_w = np.zeros(model.weights["W" + str(len(model.layers) - 1)].shape)

    momentum_1_b = np.zeros(model.weights["b" + str(len(model.layers) - 1)].shape)
    momentum_2_b = np.zeros(model.weights["b" + str(len(model.layers) - 1)].shape)

    tic = time.perf_counter()
    iteration_reached = max_epochs
    for i in range(0, max_epochs):

        batch_norms = []
        loss = []
        for Xi, Yi in batch(X, labels, batch_size):  # get batch of x and y
            # forward propagatiom
            output, forward_cache = __forward_pass(Xi, model.weights, model.layers, True)

            # backward propagation
            deltaW, deltab = __backward_pass_extreme(output, np.array(Yi), forward_cache, model.layers)

            deltaW -= (2*hyperparameters["lambda"] * model.weights["W2"])

            # print("calcolo deltaW dalla forward =\n" + str(momentum_1_w_cap))

            # update moment estimates
            momentum_1_w = (beta_1 * momentum_1_w) + ((1 - beta_1) * deltaW)
            momentum_2_w = (beta_2 * momentum_2_w) + ((1 - beta_2) * (deltaW ** 2))
            momentum_1_b = (beta_1 * momentum_1_b) + ((1 - beta_1) * deltab)
            momentum_2_b = (beta_2 * momentum_2_b) + ((1 - beta_2) * (deltab ** 2))


            # compute bias correction
            momentum_1_w_cap = momentum_1_w / (1 - (beta_1 ** (i + 1)))
            momentum_2_w_cap = momentum_2_w / (1 - (beta_2 ** (i + 1)))
            momentum_1_b_cap = momentum_1_b / (1 - (beta_1 ** (i + 1)))
            momentum_2_b_cap = momentum_2_b / (1 - (beta_2 ** (i + 1)))

            # update weight values
            model.weights["W2"] += hyperparameters["stepsize"] * (((momentum_1_w_cap) / (np.sqrt(momentum_2_w_cap) + epsilon_adam)))

            #model.weights["b2"] += hyperparameters["stepsize"] * (((momentum_1_b_cap) / (np.sqrt(momentum_2_b_cap) + epsilon_adam)))


            # save norm of the gradient for each batch
            norm_grad = LA.norm(np.array(deltaW).flatten())
            batch_norms.append(norm_grad)
            # save losses by mean in batch
            score = np.sum(np.square(np.array(model.weights["W2"]).flatten()))
            loss.append(model.score_mse(X, labels) + hyperparameters["lambda"] * score)

        norm_of_gradients.append(np.mean(batch_norms))
        variances.append(np.var(batch_norms))
        losses.append(np.mean(loss))

        # save mse or mee
        if model.problem == "classification":
            error_train.append(model.score_mse(X, labels))
            if X_validation is not None:
                error_validation.append(model.score_mse(X_validation, labels_validation))
            accuracy_train.append(model.score_accuracy(X, labels))
            if X_validation is not None:
                accuracy_validation.append(model.score_accuracy(X_validation, labels_validation))
        elif model.problem == "regression":
            error_train.append(model.score_mee(X, labels))
            if X_validation is not None:
                error_validation.append(model.score_mee(X_validation, labels_validation))
        else:
            raise Exception("Wrong problem statemenet (regression or classification)")

        if losses[-1] <= hyperparameters["epsilon"]:
            print("Stopping condition reached at iteration", i)
            print("Iteration:", i, "Loss:",
                  "{:.2e}".format(losses[-1]), "norm:", "{:.1e}".format(norm_of_gradients[-1]))
            iteration_reached = i
            break
        else:
            print("Iteration:", i, "Loss:",
                  "{:.2e}".format(losses[-1]), "norm:", "{:.1e}".format(norm_of_gradients[-1]))

        if shuffle:
            X, labels = unison_shuffle(X, labels)

        # print("\nEpoch number " + str(i) + "\n->Error:", mse_train[i])

    toc = time.perf_counter()
    print("\n")
    print(f"Seconds elapsed: {toc - tic:0.4f}")

    history["error_train"] = error_train
    history["error_validation"] = error_validation
    history["acc_train"] = accuracy_train
    history["acc_validation"] = accuracy_validation

    print("Final norm of gradient:", "{:.1e}".format(norm_of_gradients[-1]))

    # Convergence rate
    convergence_rate = []
    for i in range(0, iteration_reached):
        numerator = losses[i] - hyperparameters["epsilon"]
        denominator = hyperparameters["epsilon"]
        convergence_rate.append(numerator/denominator)

    fontsize_legend_axis = 14
    plt.plot(convergence_rate, 'r-')
    plt.title('')
    plt.ylabel('error')
    plt.yscale('log')
    plt.xlabel('iteration')
    plt.xticks(fontsize=fontsize_legend_axis)
    plt.yticks(fontsize=fontsize_legend_axis)
    plt.grid()
    plt.show()

    text_file = open("Output.txt", "w")
    for element in convergence_rate:
        text_file.write(str(element) + ",")
    text_file.close()

    fontsize_legend_axis = 14
    plt.plot(norm_of_gradients)
    plt.title('Norm of gradients')
    plt.ylabel('Norm')
    plt.yscale('log')
    plt.xlabel('iteration')
    plt.xticks(fontsize=fontsize_legend_axis)
    plt.yticks(fontsize=fontsize_legend_axis)
    plt.grid()
    plt.show()


    return history


# main used for test output
if __name__ == "__main__":
    print("Adam function tests")

    Y = np.array([[1e-8, 4e-9], [1e-10, 9e-7]])

    print(1 - (0.9 ** 2))
    print(np.sqrt(Y))
