import csv
import numpy as np

from Sources.neural_network import NeuralNetwork
from Sources.tools.load_dataset import load_monk
from itertools import product
from Sources.tools.preprocessing import one_hot
from Sources.tools.useful import k_fold


def grid_search(X, Y, hyperparameters: dict, fold_number: int, epochs:int):

    configurations = [dict(zip(hyperparameters, v)) for v in product(*hyperparameters.values())]
    results = []
    for config in configurations:
        print("Testing configuration", config)

        X = one_hot(X)

        (x_train, y_train), (x_validation, y_validation) = k_fold(X, Y, fold_number)

        # to mean result
        mse_train = []
        mse_validation = []
        accuracy_train = []
        accuracy_validation = []

        for fold in range(1, fold_number):

            # build the network
            nn = NeuralNetwork({'seed': 0,
                                'layers': [
                                    {"neurons": len(x_train[fold][0]), "activation": "linear"},
                                    # input only for dimension, insert linear
                                    {"neurons": config["neurons"], "activation": "tanh"},
                                    {"neurons": 1, "activation": "tanh"}  # output
                                ],
                                'solver': 'adam',
                                "problem": "classification"
                                })

            nn.fit(X=x_train[fold],
                   labels=[[i] for i in y_train[fold]],
                   X_validation=x_validation[fold],
                   labels_validation=[[i] for i in y_validation[fold]],
                   hyperparameters={"lambda": config["lambda"],
                                    "stepsize": config["stepsize"],
                                    "momentum": config["momentum"],
                                    "epsilon": 0.0001},
                   epochs=epochs,
                   batch_size=32,
                   shuffle=True)

            # to visualize plot for each configuration test
            nn.plot_graph()
            input()

            # store results
            mse_train.append(nn.history["mse_train"][-1])
            mse_validation.append(nn.history["mse_validation"][-1])
            accuracy_train.append(nn.history["acc_train"][-1])
            accuracy_validation.append(nn.history["acc_validation"][-1])

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


if __name__ == "__main__":
    grid_parameters = {"lambda": [0, 0.001],
                       "stepsize": [1, 0.5],
                       "momentum": [0, 0.5],
                       "neurons": [5, 10]
                       }

    (X_train, y_train, names_train), (X_test, y_test, names_test) = load_monk(2)

    grid_search(X_train, y_train, grid_parameters, fold_number=5, epochs=400)

    X_train = one_hot(X_train)

    # TODO remember to train the final model over train+validation
