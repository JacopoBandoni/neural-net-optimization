from Sources.tools.load_dataset import load_monk


if __name__ == "__main__":

    (X_train, y_train, names_train), (X_test, names_test) = load_monk(1)

    print(X_train)