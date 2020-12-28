import numpy as np


def load_monk(version: int):
    """
    Load dataset of monk in Data folder.
    Monk 1 = (a1=a2 or a5=1)
    Monk 2 = (two of a1=1, a2=1, a3=1, a4=1, a5=1, a6=1)
    Monk 3 = (a5=3, a4=1 or a5 != 4 and a2 != 3. Noisy data)
    Attribute information:
    0. class: 0, 1
    1. a1:    1, 2, 3
    2. a2:    1, 2, 3
    3. a3:    1, 2
    4. a4:    1, 2, 3
    5. a5:    1, 2, 3, 4
    6. a6:    1, 2
    7. Id:    (A unique symbol for each instance)
    :return:

    """
    if version == 1:
        print("Loading monk 1 dataset")
        path_train = '../../Data/monks-1.train'
        path_test = '../../Data/monks-1.test'
    elif version == 2:
        print("Loading monk 2 dataset")
        path_train = '../../Data/monks-2.train'
        path_test = '../../Data/monks-2.test'
    elif version == 3:
        print("Loading monk 3 dataset")
        path_train = '../../Data/monks-3.train'
        path_test = '../../Data/monks-3.test'
    else:
        raise Exception("This monk version doesn't exist")

    # load train data
    X_train = []
    y_train = []
    names_train = []
    with open(path_train, 'r') as f:
        for line in f:

            line = line.split()
            y_train.append(int(line[0]))
            X_train.append(np.array(line[1:7]).astype(int))
            names_train.append(line[7])

    if len(X_train) != len(y_train):
        raise Exception("Train Loading error")

    # load test data
    X_test = []
    y_test = []
    names_test = []
    with open(path_test, 'r') as f:
        for line in f:

            line = line.split()
            y_test.append(int(line[0]))
            X_test.append(np.array(line[1:7]).astype(int))
            names_test.append(line[7])

    if len(X_train) != len(y_train):
        raise Exception("Test Loading error")

    print("Training data:", len(X_train), ", Test data:", len(X_test))
    return (X_train, y_train, names_train), (X_test, y_test, names_test)


def load_cup20():
    print("Loading cup dataset")
    pass


# main used for test output
if __name__ == "__main__":
    print("Load dataset test")

    (X_train, y_train, names_train), (X_test, y_test, names_test) = load_monk(3)

    print(X_train)
