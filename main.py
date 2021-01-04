import csv
import numpy as np

from Sources.neural_network import NeuralNetwork
from Sources.tools.load_dataset import load_monk
from itertools import product
from Sources.tools.preprocessing import one_hot
from Sources.tools.score_function import classification_accuracy


def grid_search_k_fold(X_train, Y_train, hyperparameters: dict, fold_number: int):
    configurations = [dict(zip(hyperparameters, v)) for v in product(*hyperparameters.values())]

    results = []
    for config in configurations:
        print("Testing configuration", config)

        X_train = one_hot(X_train)

        # dividing dataset
        partition_len = int(len(X_train) / fold_number)
        rest_of_patterns = len(X_train) % fold_number
        X_partitioned = [X_train[i:i + partition_len] for i in range(0, len(X_train)-rest_of_patterns, partition_len)]
        Y_partitioned = [Y_train[i:i + partition_len] for i in range(0, len(Y_train)-rest_of_patterns, partition_len)]

        # to mean result
        mse_train = []
        mse_validation = []
        accuracy_train = []
        accuracy_validation = []

        for fold in range(0, fold_number):
            # creating partition mutually exclusive
            x_subset = X_partitioned[:fold] + X_partitioned[fold+1:]
            x_train = np.concatenate(x_subset)

            y_subset = Y_partitioned[:fold] + Y_partitioned[fold+1:]
            y_train = np.concatenate(y_subset)

            x_validation = X_partitioned[fold]
            y_validation = Y_partitioned[fold]

            # train the network over set
            nn = NeuralNetwork({'seed': 0,
                                'layers': [
                                    {"neurons": len(x_train[0]), "activation": "linear"},
                                    # input only for dimension, insert linear
                                    {"neurons": config["neurons"], "activation": "sigmoid"},
                                    {"neurons": 1, "activation": "sigmoid"}  # output
                                ],
                                'solver': 'adam',
                                "problem": "classification"
                                })

            nn.fit(X=x_train,
                   labels=[[i] for i in y_train],
                   X_validation=x_validation,
                   labels_validation=[[i] for i in y_validation],
                   hyperparameters={"lambda": config["lambda"],
                                    "stepsize": config["stepsize"],
                                    "momentum": config["momentum"],
                                    "epsilon": 0.0001},
                   epochs=10)

            # store results
            mse_train.append(nn.score(x_train, y_train))
            mse_validation.append(nn.score(x_validation, y_validation))

            treshold_list_train = [1 if i > 0.5 else 0 for i in nn.predict(x_train)]
            treshold_list_test = [1 if i > 0.5 else 0 for i in nn.predict(x_validation)]

            accuracy_train.append(classification_accuracy(treshold_list_train, y_train))
            accuracy_validation.append(classification_accuracy(treshold_list_test, y_validation))

        experiment_data = {}
        for name in config:
            experiment_data[name] = config[name]
        # compute mean over fold
        experiment_data["mse_train"] = np.mean(mse_train)
        experiment_data["mse_test"] = np.mean(mse_validation)

        experiment_data["acc_train"] = np.mean(accuracy_train)
        experiment_data["acc_test"] = np.mean(accuracy_validation)

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
    grid_parameters = {"lambda": [0.1, 0.3],
                       "stepsize": [0.1, 0.3],
                       "momentum": [0.5, 0.7],
                       "neurons": [10, 50]
                       }

    (X_train, y_train, names_train), (X_test, y_test, names_test) = load_monk(2)

    grid_search_k_fold(X_train, y_train, grid_parameters, fold_number=5)

    # TODO remember to train the final model over train+validation
