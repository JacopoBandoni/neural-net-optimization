import csv
import numpy as np

from Sources.neural_network import NeuralNetwork
from Sources.tools.load_dataset import load_monk, load_cup20
from itertools import product
from Sources.tools.preprocessing import one_hot
from Sources.tools.useful import k_fold, grid_search, hold_out

if __name__ == "__main__":
    grid_parameters = {"lambda": [0.00, 0.00005],
                       "stepsize": [0.0001, 0.001],
                       "momentum": [0.0, 0.2],
                       "epsilon": [0.0009],
                       "batch_size": [32, 128], # mini-batch vs online
                       # insert number of HIDDEN layer where you will insert hyperparams
                       "layer_number": [1, 5],
                       # for each layer the element to test
                       "neuron": [30],
                       "activation": ["tanh", "sigmoid"],
                       "activation_output": ["linear"],
                       "initialization": ["uniform", "xavier"]
                       }

    epochs = 2000

        # load dataset
    (X_train, y_train, names_train), (X_test, names_test) = load_cup20()
    # if is classification X_train = one_hot(X_train)

    # TODO sul cup FARE HOLDOUT per avere un TS interno, necessario per i plots (keep percentage as original split)
    X_train, y_train, X_test, y_test = hold_out(X_train, y_train, percentage=25)
    print("New Training data:", len(X_train), ", New test data:", len(X_test))

    # load configurations to test
    configurations = grid_search(grid_parameters)

    # for each configuration produced by grid search build and train model over k fold
    results = []
    for count, config in enumerate(configurations):
        
        print("Testing configuration", count, "(", len(configurations), "):", config)
        # produce set mutually exclusive
        X_T, Y_T, X_V, Y_V = k_fold(X_train, y_train, fold_number=3)

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
                elif dim == config["layer_number"]+1:   # last layer (2 for cup)
                    topology.append({"neurons": 2, "activation": config["activation_output"]})
                else:   # hidden layers
                    topology.append({"neurons": config["neuron"], "activation": config["activation"]})

            # build and train the network
            nn = NeuralNetwork({'seed': 0,
                                'layers': topology,
                                'solver': 'adam',
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

            # to visualize plot for each configuration test
            nn.plot_graph()
            input()

            # store results
            mse_train.append(nn.history["error_train"][-1])
            mse_validation.append(nn.history["error_validation"][-1])
            # accuracy_train.append(nn.history["acc_train"][-1])
            # accuracy_validation.append(nn.history["acc_validation"][-1])

        # over k fold compute mean
        experiment_data = {}
        for name in config:
            experiment_data[name] = config[name]
        # compute mean over fold
        experiment_data["error_train"] = np.mean(mse_train)
        experiment_data["error_test"] = np.mean(mse_validation)
        experiment_data["error_train_variance"] = np.var(mse_train)
        experiment_data["error_test_variance"] = np.var(mse_validation)
        """
        experiment_data["acc_train"] = np.mean(accuracy_train)
        experiment_data["acc_test"] = np.mean(accuracy_validation)
        experiment_data["acc_train_variance"] = np.var(accuracy_train)
        experiment_data["acc_validation_variance"] = np.var(accuracy_validation)
        """
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
            print("\nSaved file:", filename)
    except IOError:
        print("Csv writing error")

