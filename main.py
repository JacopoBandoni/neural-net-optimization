from Sources.tools.load_dataset import load_monk


def grid_search(X, labels, hyperparameters:dict):

    print(hyperparameters.keys())




if __name__=="__main__":
    print("hello world")

    hyperparameters = {"lambda": 0,
                       "stepsize": 0.5,
                       "momentum": 1,
                       "epsilon": 0.001}

    X, Y = load_monk(2)

    grid_search(1, 2, hyperparameters)
