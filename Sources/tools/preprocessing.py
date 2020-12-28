import numpy as np

def one_hot (values):
    """
    :param values: The whole dataset
    :returns the dataset in one-hot value encoding, it is assumed that the possible values
             for the attributes range from 1 to the maximum value of each attribute
    """

    #cambiare inserendo lista con il max e il min valore di ogni attributo

    max_values = np.max(np.array(values), axis=0)
    one_hot_value = []
    one_hot_values = []

    for z in range(0, len(values)):  # ciclo per ogni sample dell'insieme di sample
        for i in range(0, len(values[0])):  # ciclo per ogni attributo
            for j in range(1, max_values[i]+1):  # ciclo per ogni one_hot value di ogni attributo
                if values[z][i] == j:
                    one_hot_value.insert(len(one_hot_value), 1)
                else:
                    one_hot_value.insert(len(one_hot_value), 0)

        one_hot_values.insert(len(one_hot_values), one_hot_value)
        one_hot_value = []

    return one_hot_values

if __name__ == "__main__":
    lista = [[1, 2, 3], [2, 3, 0]]
    print(one_hot(lista))
