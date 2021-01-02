import csv

from Sources.neural_network import NeuralNetwork
from Sources.tools.load_dataset import load_monk
from itertools import product
from Sources.tools.preprocessing import one_hot
from Sources.tools.score_function import classification_accuracy


def grid_search(X, Y, hyperparameters: dict):
    configurations = [dict(zip(hyperparameters, v)) for v in product(*hyperparameters.values())]

    results = []
    for config in configurations:
        print("Testing configuration", config)
        nn = NeuralNetwork({'seed': 0,
                            'layers': [
                                {"neurons": len(one_hot(X[0])[0]), "activation": "linear"},
                                # input only for dimension, insert linear
                                {"neurons": config["neurons"], "activation": "sigmoid"},
                                {"neurons": 1, "activation": "sigmoid"}  # output
                            ],
                            'solver': 'sgd',
                            "problem": "classification"
                            })

        nn.fit(X=one_hot(X[0]),
               labels=[[i] for i in X[1]],
               hyperparameters={"lambda": config["lambda"],
                                "stepsize": config["stepsize"],
                                "momentum": config["momentum"],
                                "epsilon": 0.0001},
               epochs=10)

        experiment_data = {}
        for name in config:
            experiment_data[name] = config[name]

        experiment_data["mse_train"] = nn.score(X=one_hot(X[0]), labels=[[i] for i in X[1]])
        experiment_data["mse_test"] = nn.score(X=one_hot(Y[0]), labels=[[i] for i in Y[1]])

        treshold_list_train = [1 if i > 0.5 else 0 for i in nn.predict(one_hot(X[0]))]
        treshold_list_test = [1 if i > 0.5 else 0 for i in nn.predict(one_hot(Y[0]))]
        experiment_data["acc_train"] = classification_accuracy(output=treshold_list_train, target=X[1])
        experiment_data["acc_test"] = classification_accuracy(output=treshold_list_test, target=Y[1])

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
    grid_parameters = {"lambda": [0.1,  0.3],
                       "stepsize": [0.1, 0.3],
                       "momentum": [0.5, 0.7],
                       "neurons": [10, 50]
                       }

    X, Y = load_monk(2)

    grid_search(X, Y, grid_parameters)

