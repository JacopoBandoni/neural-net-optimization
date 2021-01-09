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
    X_T, Y_T, X_V, Y_V = k_fold(X_train, y_train, fold_number=10)

    # to mean result
    mse_train = []
    mse_validation = []
    accuracy_train = []
    accuracy_validation = []
    for (x_t, y_t, x_v, y_v) in zip(X_T, Y_T, X_V, Y_V):
        # build and train the network
        nn = NeuralNetwork({'seed': np.random.randint(100),
                            'layers': [
                                {"neurons": len(x_t[0]), "activation": "linear"},
                                {"neurons": 5, "activation": "tanh"},
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
               hyperparameters={"lambda": 0.000,
                                "stepsize": 0.9,
                                "momentum": 0.2,
                                "epsilon": 0.0001
                                },
               epochs=500, batch_size=32, shuffle=True)

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

    # ---------------------------------------------------
    # Model Assesment
    print(" ----- MODEL ASSESMENT ----- ")
    input()
    trials = 10

    # y must be a column vector, not row one
    y_train = [[i] for i in y_train]
    y_test = [[i] for i in y_test]
    # TODO convert test to one hot ?
    X_test = one_hot(X_test)

    mse_train = []
    mse_test = []
    accuracy_train = []
    accuracy_test = []
    for t in range(trials):
        # build and train the network
        nn = NeuralNetwork({'seed': np.random.randint(100),
                            'layers': [
                                {"neurons": len(X_train[0]), "activation": "linear"},
                                {"neurons": 5, "activation": "tanh"},
                                {"neurons": 1, "activation": "tanh"}
                            ],
                            'solver': 'sgd',
                            "problem": "classification",
                            "initialization": "uniform"
                            })

        nn.fit(X=X_train, labels=y_train,
               X_validation=X_test, labels_validation=y_test,
               hyperparameters={"lambda": 0.000,
                                "stepsize": 0.9,
                                "momentum": 0.2,
                                "epsilon": 0.0001
                                },
               epochs=500, batch_size=32, shuffle=True)

        # store results
        mse_train.append(nn.history["mse_train"][-1])
        mse_test.append(nn.history["mse_validation"][-1])
        accuracy_train.append(nn.history["acc_train"][-1])
        accuracy_test.append(nn.history["acc_validation"][-1])

    print("MSE TR:", np.mean(mse_train))
    print("MSE TR variance:", np.var(mse_train))
    print("MSE TS:", np.mean(mse_test))
    print("MSE TS variance:", np.var(mse_test))
    print("ACC TR:", np.mean(accuracy_train))
    print("ACC TR variance:", np.var(accuracy_train))
    print("ACC TS:", np.mean(accuracy_test))
    print("ACC TS variance:", np.var(accuracy_test))
