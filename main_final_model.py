from Sources.neural_network import NeuralNetwork
from Sources.tools.load_dataset import load_monk, load_cup20
from Sources.tools.preprocessing import one_hot
from Sources.tools.useful import k_fold, hold_out

import numpy as np

if __name__ == "__main__":
    seed = 0
    # load data
    (X_train, y_train, names_train), (X_test, names_test) = load_cup20()
    # hold out
    X_train, y_train, X_test_new, y_test_new = hold_out(X_train, y_train, percentage=25)
    print("New Training data:", len(X_train), ", New test data:", len(X_test_new))

    # produce set mutually exclusive
    X_T, Y_T, X_V, Y_V = k_fold(X_train, y_train, fold_number=10)

    # to mean result
    error_train = []
    error_validation = []
    for (x_t, y_t, x_v, y_v) in zip(X_T, Y_T, X_V, Y_V):
        # build and train the network
        nn = NeuralNetwork({'seed': 0,
                            'layers': [
                                {"neurons": len(x_t[0]), "activation": "linear"},
                                {"neurons": 20, "activation": "sigmoid"},
                                {"neurons": 20, "activation": "sigmoid"},
                                {"neurons": 20, "activation": "sigmoid"},
                                {"neurons": 2, "activation": "linear"}
                            ],
                            'solver': 'sgd',
                            "problem": "regression",
                            "initialization": "xavier"
                            })

        nn.fit(X=x_t, labels=y_t,
               X_validation=x_v, labels_validation=y_v,
               hyperparameters={"lambda": 0.00,
                                "stepsize": 0.0005,
                                "momentum": 0.000,
                                "epsilon": 0.0001
                                },
               epochs=4000, batch_size=64, shuffle=True)

        # to visualize plot for each configuration test
        nn.plot_graph()
        input()

        # store results
        error_train.append(nn.history["error_train"][-1])
        error_validation.append(nn.history["error_validation"][-1])

    print("MSE TR:", np.mean(error_train))
    print("MSE TR variance:", np.var(error_train))
    print("MSE VL:", np.mean(error_validation))
    print("MSE VL variance:", np.var(error_validation))

    # ---------------------------------------------------
    # Model Assesment
    print(" ----- MODEL ASSESMENT ----- ")
    input()
    trials = 10

    mse_train = []
    mse_test = []
    for t in range(trials):
        # build and train the network
        nn = NeuralNetwork({'seed': seed,
                            'layers': [
                                {"neurons": len(X_train[0]), "activation": "linear"},
                                {"neurons": 100, "activation": "tanh"},
                                {"neurons": 1, "activation": "linear"}
                            ],
                            'solver': 'extreme_adam',
                            "problem": "classification",
                            "initialization": "uniform"
                            })

        nn.fit(X=X_train, labels=y_train,
               X_validation=X_test_new, labels_validation=y_test_new,
               hyperparameters={"lambda": 0.005,
                                "stepsize": 0.0009,
                                # "momentum": 0.2,
                                "epsilon": 0.0001
                                },
               epochs=1000, batch_size=32, shuffle=True)

        # store results
        mse_train.append(nn.history["mse_train"][-1])
        mse_test.append(nn.history["mse_validation"][-1])

    print("MSE TR:", np.mean(mse_train))
    print("MSE TR variance:", np.var(mse_train))
    print("MSE TS:", np.mean(mse_test))
    print("MSE TS variance:", np.var(mse_test))

