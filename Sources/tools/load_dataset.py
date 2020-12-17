def load_monk_1():
    """
    Load dataset of monk (a1=a2 or a5=1) in Data folder. Attribute information:
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
    print("Loading monk 1 dataset")

    # load train data
    X_train = []
    y_train = []
    names_train = []
    with open('../../Data/monks-1.train', 'r') as f:
        for line in f:
            line = line.split()

            y_train.append(line[0])
            X_train.append(line[1:7])
            names_train.append(line[7])

    if len(X_train) != len(y_train):
        raise Exception("Train Loading error")

    # load test data
    X_test = []
    y_test = []
    names_test = []
    with open('../../Data/monks-1.test', 'r') as f:
        for line in f:
            line = line.split()

            y_test.append(line[0])
            X_test.append(line[1:7])
            names_test.append(line[7])

    if len(X_train) != len(y_train):
        raise Exception("Test Loading error")

    print("Training data:", len(X_train), ", Test data:", len(X_test))
    return (X_train, y_train, names_train), (X_test, y_test, names_test)


def load_monk_2():
    """
        Load dataset of monk (two of a1=1, a2=1, a3=1, a4=1, a5=1, a6=1) in Data folder. Attribute information:
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
    print("Loading monk 2 dataset")

    # load train data
    X_train = []
    y_train = []
    names_train = []
    with open('../../Data/monks-2.train', 'r') as f:
        for line in f:
            line = line.split()

            y_train.append(line[0])
            X_train.append(line[1:7])
            names_train.append(line[7])

    if len(X_train) != len(y_train):
        raise Exception("Train Loading error")

    # load test data
    X_test = []
    y_test = []
    names_test = []
    with open('../../Data/monks-2.test', 'r') as f:
        for line in f:
            line = line.split()

            y_test.append(line[0])
            X_test.append(line[1:7])
            names_test.append(line[7])

    if len(X_train) != len(y_train):
        raise Exception("Test Loading error")

    print("Training data:", len(X_train), ", Test data:", len(X_test))
    return (X_train, y_train, names_train), (X_test, y_test, names_test)


def load_monk_3():
    """
        Load dataset of monk 3 (a5=3, a4=1 or a5 != 4 and a2 != 3. Noisy data) stored in Data folder.
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
    print("Loading monk 3 dataset")

    # load train data
    X_train = []
    y_train = []
    names_train = []
    with open('../../Data/monks-3.train', 'r') as f:
        for line in f:
            line = line.split()

            y_train.append(line[0])
            X_train.append(line[1:7])
            names_train.append(line[7])

    if len(X_train) != len(y_train):
        raise Exception("Train Loading error")

    # load test data
    X_test = []
    y_test = []
    names_test = []
    with open('../../Data/monks-3.test', 'r') as f:
        for line in f:
            line = line.split()

            y_test.append(line[0])
            X_test.append(line[1:7])
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

    load_monk_3()
