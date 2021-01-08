from Sources.neural_network import NeuralNetwork
from Sources.tools.load_dataset import load_monk
from Sources.tools.preprocessing import one_hot
from Sources.tools.useful import k_fold

import numpy as np

if __name__ == "__main__":

    # load dataset
    (X_train, y_train, names_train), (X_test, y_test, names_test) = load_monk(1)
    # if is classification
    X_train = one_hot(X_train)

    # produce set mutually exclusive
    X_T, Y_T, X_V, Y_V = k_fold(X_train, y_train, fold_number=3)

    # to mean result
    mse_train = []
    mse_validation = []
    accuracy_train = []
    accuracy_validation = []
    for (x_t, y_t, x_v, y_v) in zip(X_T, Y_T, X_V, Y_V):
        # build and train the network
        nn = NeuralNetwork({'seed': 0,
                            'layers': [
                                {"neurons": len(x_t[0]), "activation": "linear"},
                                {"neurons": 5, "activation": "sigmoid"},
                                {"neurons": 5, "activation": "sigmoid"},
                                {"neurons": 1, "activation": "tanh"}
                            ],
                            'solver': 'sgd',
                            "problem": "classification",
                            "initialization": "uniform"
                            })
        # y must be a column vector, not row one
        y_t = [[i] for i in y_t]
        y_v = [[i] for i in y_v]

        nn.fit(X=x_t, labels=y_t,
               X_validation=x_v, labels_validation=y_v,
               hyperparameters={"lambda": 0.01,
                                "stepsize": 0.3,
                                "momentum": 0.5,
                                "epsilon": 0.0001
                                },
               epochs=200, batch_size=len(x_t), shuffle=True)

        # to visualize plot for each configuration test
        nn.plot_graph()
        input()

        # store results
        mse_train.append(nn.history["mse_train"][-1])
        mse_validation.append(nn.history["mse_validation"][-1])
        accuracy_train.append(nn.history["acc_train"][-1])
        accuracy_validation.append(nn.history["acc_validation"][-1])

    print("MSE TR:", np.mean(mse_train))
    print("MSE TR variance:", np.var(mse_train))
    print("MSE VL:", np.mean(mse_validation))
    print("MSE VL variance:", np.var(mse_validation))
    print("ACC TR:", np.mean(accuracy_train))
    print("ACC TR variance:", np.var(accuracy_train))
    print("ACC VL:", np.mean(accuracy_validation))
    print("ACC VL variance:", np.var(accuracy_validation))

