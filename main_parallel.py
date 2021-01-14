import csv
import numpy as np

from Sources.neural_network import NeuralNetwork
from Sources.tools.load_dataset import load_monk, load_cup20
from itertools import product
from Sources.tools.preprocessing import one_hot
from Sources.tools.useful import k_fold, grid_search, hold_out
from joblib import Parallel, delayed


def grid_test(config, X_train, y_train):
    print("Testing configuration: ", config)
    # produce set mutually exclusive
    X_T, Y_T, X_V, Y_V = k_fold(X_train, y_train, fold_number=3)

    # to mean result
    error_train = []
    error_validation = []
    for (x_t, y_t, x_v, y_v) in zip(X_T, Y_T, X_V, Y_V):

        # function to build topology
        topology = []
        for dim in range(0, config["layer_number"] + 2):
            if dim == 0:  # first layer
                topology.append({"neurons": len(x_t[0]), "activation": "linear"})
            elif dim == config["layer_number"] + 1:  # last layer (2 for cup)
                topology.append({"neurons": 2, "activation": config["activation_output"]})
            else:  # hidden layers
                topology.append({"neurons": config["neuron"], "activation": config["activation"]})

        # build and train the network
        nn = NeuralNetwork({'seed': 0,
                            'layers': topology,
                            'solver': 'extreme_adam',
                            "problem": "regression",
                            "initialization": config["initialization"]
                            })

        # y must be a column vector, not row one
        # classification y_t = [[i] for i in y_t]
        # classification y_v = [[i] for i in y_v]

        nn.fit(X=x_t, labels=y_t,
               X_validation=x_v, labels_validation=y_v,
               hyperparameters={"lambda": config["lambda"],
                                "stepsize": config["stepsize"],
                                "momentum": config["momentum"],
                                "epsilon": config["epsilon"]},
               epochs=epochs, batch_size=config["batch_size"], shuffle=True)

        # nn.plot_graph()
        # input()

        # store results
        error_train.append(nn.history["error_train"][-1])
        error_validation.append(nn.history["error_validation"][-1])

    experiment_data = {}
    for name in config:
        experiment_data[name] = config[name]
    # compute mean over fold
    experiment_data["error_train"] = np.mean(error_train)
    experiment_data["error_test"] = np.mean(error_validation)
    experiment_data["error_train_variance"] = np.var(error_train)
    experiment_data["error_test_variance"] = np.var(error_validation)

    return experiment_data


if __name__ == "__main__":
    grid_parameters = {"lambda": [0.00, 0.0005],
                       "stepsize": [0.001, 0.01, 0.07],
                        "momentum": [0.0],
                       "epsilon": [0.0009],
                       "batch_size": [64, 128],  # mini-batch vs online
                       # insert number of HIDDEN layer where you will insert hyperparams
                       "layer_number": [1],
                       # for each layer the element to test
                       "neuron": [30, 80, 200],
                       "activation": ["tanh", "sigmoid"],
                       "activation_output": ["linear"],
                       "initialization": ["uniform", "xavier"]
                       }

    epochs = 2000

    # load data
    (X_train, y_train, names_train), (X_test, names_test) = load_cup20()
    # hold out
    X_train, y_train, X_test, y_test = hold_out(X_train, y_train, percentage=25)
    print("New Training data:", len(X_train), ", New test data:", len(X_test))

    configurations = grid_search(grid_parameters)
    print("Number of configurations", len(configurations))

    results = Parallel(n_jobs=6, verbose=10)(delayed(grid_test)(config, X_train, y_train) for config in configurations)

    # save results to csv file
    csv_columns = results[0].keys()
    filename = "results.csv"
    try:
        with open(filename, 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
            writer.writeheader()
            for experiment in results:
                writer.writerow(experiment)
            print("\nSaved file:", filename)
    except IOError:
        print("Csv writing error")
