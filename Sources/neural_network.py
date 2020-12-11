import numpy as np
"""
Neural network with one hidden layer, y=W2σ(W1x), where:
the weight matrix for the hidden layer W1 is a fixed random matrix,
σ(⋅) is an elementwise activation function of your choice,
the output weight matrix W2 is chosen by solving a linear least-squares problem (with L_2 regularization)
"""


class ExtremeLearningMachine():
    def __init__(self, n_hidden_units=0,  activation="None", seed=0, dim={}):

        np.random.seed(seed)    # fix seed for test reproducibility

        # classic argument check
        if n_hidden_units == 0:
            raise Exception("You should pass an hidden units number")
        if activation == "None":
            raise  Exception("You should pass an activation function")
        if type(dim) is not dict:
            raise Exception("You should pass an input dimension")
        if len(dim) == 2:    # check correctness of dimension

        else:
            raise Exception("You passed as dimension more variables thant needed")


        self.n_hidden_units = n_hidden_units
        self.W1 = np.random.randn()
        self.W2 = np.random.randn()
        self.activation = activation



