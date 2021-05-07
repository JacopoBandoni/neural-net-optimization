from Sources.neural_network import NeuralNetwork
from Sources.tools.load_dataset import load_monk
from Sources.tools.preprocessing import one_hot
from Sources.tools.useful import unison_shuffle
import time

if __name__ == "__main__":

    (X_train, y_train, names_train), (X_test, y_test, names_test) = load_monk(1)

    X_test, y_test = unison_shuffle(X_test, y_test)

    X_test = one_hot(X_test)
    y_test = [[i] for i in y_test]

    print("X Matrix dimension:", len(X_test), "x", len(X_test[1]))
    print("Y matrix dimension:", len(y_test), "x", len(y_test[1]))

    # adam last lineare (hidden a scelta)
    layers = [{"neurons": len(X_test[1]), "activation": "linear"},
              {"neurons": 200, "activation": "tanh"},
              {"neurons": 1, "activation": "linear"}]

    # ADAM meglio batch e non batch

    # build and train the network
    nn = NeuralNetwork({'seed': 0,
                        'layers': layers,
                        'solver': 'extreme_adam',
                        "problem": "classification",
                        "initialization": "uniform"
                        })

    nn.fit(X=X_test, labels=y_test,
           X_validation=None, labels_validation=None,
           hyperparameters={"lambda": 0.1, # test also lambda in table
                            "stepsize": 0.01,
                            #"momentum": "None",
                            #"epsilon": 0.000
                            },
           epochs=200, batch_size=len(X_test), shuffle=False)

    # nn.plot_graph()