import csv
import numpy as np

from Sources.neural_network import NeuralNetwork
from Sources.tools.load_dataset import load_monk
from itertools import product
from Sources.tools.preprocessing import one_hot
from Sources.tools.useful import k_fold, grid_search

if __name__ == "__main__":
    grid_parameters = {"lambda": [0.0],
                       "stepsize": [0.7],
                       "momentum": [0.0],
                       "epsilon": [0.0009],
                       "batch_size": [1], # mini-batch vs online
                       # insert number of HIDDEN layer where you will insert hyperparams
                       "layer_number": [2],
                       # for each layer the element to test
                       "neuron": [5],
                       "activation": ["tanh"],
                       "activation_output": ["tanh"],
                       "initialization": ["uniform"]
                       }

    epochs = 400

    # load dataset
    (X_train, y_train, names_train), (X_test, y_test, names_test) = load_monk(1)
    # if is classification
    X_train = one_hot(X_train)

    # load configurations to test
    configurations = grid_search(grid_parameters)
    # for each configuration produced by grid search build and train model over k fold
    results = []
    for count, config in enumerate(configurations):
        print("Testing configuration", count, "(", len(configurations), "):", config)
        # produce set mutually exclusive
        X_T, Y_T, X_V, Y_V = k_fold(X_train, y_train, fold_number=5)

        # to mean result
        mse_train = []
        mse_validation = []
        accuracy_train = []
        accuracy_validation = []
        for (x_t, y_t, x_v, y_v) in zip(X_T, Y_T, X_V, Y_V):

            # function to build topology
            topology = []
            for dim in range(0, config["layer_number"]+2):
                if dim == 0:    # first layer
                    topology.append({"neurons": len(x_t[0]), "activation": "linear"})
                elif dim == config["layer_number"]+1:   # last layer
                    topology.append({"neurons": 1, "activation": config["activation_output"]})
                else:   # hidden layers
                    topology.append({"neurons": config["neuron"], "activation": config["activation"]})

            # build and train the network
            nn = NeuralNetwork({'seed': 0,
                                'layers': topology,
                                'solver': 'sgd',
                                "problem": "classification",
                                "initialization": config["initialization"]
                                })

            # y must be a column vector, not row one
            y_t = [[i] for i in y_t]
            y_v = [[i] for i in y_v]

            nn.fit(X=x_t, labels=y_t,
                   X_validation=x_v, labels_validation=y_v,
                   hyperparameters={"lambda": config["lambda"],
                                    "stepsize": config["stepsize"],
                                    "momentum": config["momentum"],
                                    "epsilon": config["epsilon"]},
                   epochs=epochs, batch_size=config["batch_size"], shuffle=True)

            # to visualize plot for each configuration test
            nn.plot_graph()
            input()

            # store results
            mse_train.append(nn.history["mse_train"][-1])
            mse_validation.append(nn.history["mse_validation"][-1])
            accuracy_train.append(nn.history["acc_train"][-1])
            accuracy_validation.append(nn.history["acc_validation"][-1])

        # over k fold compute mean
        experiment_data = {}
        for name in config:
            experiment_data[name] = config[name]
        # compute mean over fold
        experiment_data["mse_train"] = np.mean(mse_train)
        experiment_data["mse_test"] = np.mean(mse_validation)
        experiment_data["mse_train_variance"] = np.var(mse_train)
        experiment_data["mse_test_variance"] = np.var(mse_validation)

        experiment_data["acc_train"] = np.mean(accuracy_train)
        experiment_data["acc_test"] = np.mean(accuracy_validation)
        experiment_data["acc_train_variance"] = np.var(accuracy_train)
        experiment_data["acc_validation_variance"] = np.var(accuracy_validation)

        results.append(experiment_data)

    # save results to csv file
    csv_columns = results[0].keys()
    filename = "results.csv"
    try:
        with open(filename, 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
            writer.writeheader()
            for experiment in results:
                writer.writerow(experiment)
    except IOError:
        print("Csv writing error")

