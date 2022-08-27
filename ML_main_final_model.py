import csv

from joblib import Parallel, delayed

from Sources.neural_network import NeuralNetwork
from Sources.tools.load_dataset import load_monk, load_cup20
from Sources.tools.preprocessing import one_hot
from Sources.tools.useful import k_fold, hold_out

import numpy as np


def to_parallelize(x_t, y_t, x_v, y_v):
    # build and train the network
    solver = 'adam'
    nn = NeuralNetwork({'seed': 3,
                        'layers': [
                            {"neurons": len(x_t[0]), "activation": "linear"},
                            {"neurons": 50, "activation": "sigmoid"},
                            {"neurons": 50, "activation": "sigmoid"},
                            {"neurons": 50, "activation": "sigmoid"},
                            {"neurons": 2, "activation": "linear"}
                        ],
                        'solver': solver,
                        "problem": "regression",
                        "initialization": "uniform"
                        })
    nn.fit(X=x_t, labels=y_t,
           X_validation=x_v, labels_validation=y_v,
           hyperparameters={"lambda": 0.00005,
                            "stepsize": 0.0008,
                            "momentum": "None",
                            "epsilon": 0.0009
                            },
           epochs=3000, batch_size=64, shuffle=True)
    # to visualize plot for each configuration test
    # nn.plot_graph()
    # input()

    errors = [nn.history["error_train"][-1], nn.history["error_validation"][-1]]

    return errors


if __name__ == "__main__":
    seed = 3
    """
    # load data
    (X_train, y_train, names_train), (X_test, names_test) = load_cup20()
    # hold out
    X_train, y_train, X_test_new, y_test_new = hold_out(X_train, y_train, percentage=25)
    print("New Training data:", len(X_train), ", New test data:", len(X_test_new))

    
    # produce set mutually exclusive
    X_T, Y_T, X_V, Y_V = k_fold(X_train, y_train, fold_number=6)

    errors = Parallel(n_jobs=6, verbose=10)(delayed(to_parallelize)(x_t, y_t, x_v, y_v)
                                                                   for (x_t, y_t, x_v, y_v) in zip(X_T, Y_T, X_V, Y_V))

    print("MEE TR:", np.mean(errors[:][0]))
    print("MEE TR variance:", np.var(errors[:][0]))
    print("MEE VL:", np.mean(errors[:][1]))
    print("MEE VL variance:", np.var(errors[:][1]))
    
    """

    # ---------------------------------------------------
    """
    # Model Assesment
    print(" ----- MODEL ASSESMENT ----- ")
    input()
    trials = 3

    mee_train = []
    mee_test = []
    for t in range(trials):
        solver = 'adam'
        nn = NeuralNetwork({'seed': 3,
                            'layers': [
                                {"neurons": len(X_train[0]), "activation": "linear"},
                                {"neurons": 50, "activation": "sigmoid"},
                                {"neurons": 50, "activation": "sigmoid"},
                                {"neurons": 50, "activation": "sigmoid"},
                                {"neurons": 2, "activation": "linear"}
                            ],
                            'solver': solver,
                            "problem": "regression",
                            "initialization": "uniform"
                            })
        nn.fit(X=X_train, labels=y_train,
               X_validation=X_test_new, labels_validation=y_test_new,
               hyperparameters={"lambda": 0.00005,
                                "stepsize": 0.0008,
                                "momentum": "None",
                                "epsilon": 0.0009
                                },
               epochs=3000, batch_size=64, shuffle=True)

        # nn.plot_graph()

        # store results
        mee_train.append(nn.history["error_train"][-1])
        mee_test.append(nn.history["error_validation"][-1])

    print("MSE TR:", np.mean(mee_train))
    print("MSE TR variance:", np.var(mee_train))
    print("MSE TS:", np.mean(mee_test))
    print("MSE TS variance:", np.var(mee_test))
        
    """
    # -----------------------------------------------------------------------
    # MODEL PREDICTION
    # load data
    (X_train, y_train, names_train), (X_test, names_test) = load_cup20()

    solver = 'adam'
    nn = NeuralNetwork({'seed': 3,
                        'layers': [
                            {"neurons": len(X_train[0]), "activation": "linear"},
                            {"neurons": 50, "activation": "sigmoid"},
                            {"neurons": 50, "activation": "sigmoid"},
                            {"neurons": 50, "activation": "sigmoid"},
                            {"neurons": 2, "activation": "linear"}
                        ],
                        'solver': solver,
                        "problem": "regression",
                        "initialization": "uniform"
                        })
    nn.fit(X=X_train, labels=y_train,
           X_validation=None, labels_validation=None,
           hyperparameters={"lambda": 0.00005,
                            "stepsize": 0.0008,
                            "momentum": "None",
                            "epsilon": 0.0009
                            },
           epochs=3000, batch_size=64, shuffle=True)

    print("MSE TR:", nn.history["error_train"][-1])

    outputs = nn.predict(X_test)

    filename = "output.csv"
    try:
        with open(filename, 'w') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            for name, result in zip(names_test, outputs):
                writer.writerow([name, result[0], result[1]])
            print("\nSaved file:", filename)
    except IOError:
        print("Csv writing error")