import numpy as np
"""
Neural network with one hidden layer, y=W2σ(W1x), where:
the weight matrix for the hidden layer W1 is a fixed random matrix,
σ(⋅) is an elementwise activation function of your choice,
the output weight matrix W2 is chosen by solving a linear least-squares problem (with L_2 regularization)
"""


class ExtremeLearningMachine():
    """
    Parameters mandatory inside the dictionary, recognized by the model are:
    units = [1, +inf)   number of units of the unique hidden layer
    seed = 0
    activation = tanh, sigmoid
    features = [1, +inf)    number of column each pattern has
    output = [1, +inf)      number of output neurons
    """
    def __init__(self, settings:dict):
        # classic argument check
        if "seed" in settings:
            np.random.seed(settings["seed"])    # fix seed for test reproducibility
        else:
            raise Exception("Seed not passed")

        if "units" in settings and settings["units"] > 0:
            self.units = settings["units"]
        else:
            raise Exception("Units number error")

        if "activation" in settings and (settings["activation"] == "tanh" or settings["activation"] == "sigmoid"):
            self.activation = settings["activation"]
        else:
            raise Exception("Activation function error")

        if "features" in settings and settings["features"] > 0:
            self.features = settings["features"]
        else:
            raise Exception("Column number of pattern error")

        if "output" in settings and settings["output"]> 0:
            self.output = settings["output"]
        else:
            raise Exception("Output units number error")

        self.W1 = np.random.randn(self.features, self.units)    # initialize weight fixed matrix
        self.W2 = np.random.randn(self.units, self.output)      # initialize weight to optimize

        print("Model initialized")

    """ 
    Parameters mandatory inside the fit model:
    X:  training data where shapes match with initialization values
    Labels:  training y data where shapes match with initialization values  
    """
    def fit(self, X, labels):
        # classic argument check
        for i in range(0, len(X[0])):   # for each samples check features dimension
            if len(X[i][1]) != self.features:
                raise Exception("Miss features in some pattern")

        if len(labels) != len(X[0]):
            raise Exception("Label dimension mismatch")

    """ Parameter
    X: Test data where is computed output
    """
    def predict(self, X):
        pass

    """ 
    X: test data where is computed output
    Labels: target output where is computed scored function
    """
    def score(self, X, labels):
        pass


# main used for test output
if __name__ == "__main__":
    print("Neural network tests: extreme learning")
