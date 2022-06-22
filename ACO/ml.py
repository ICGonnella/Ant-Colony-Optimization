import numpy as np


def obtain_probabilities(test, model):
    z = model.decision_function(np.array(test))
    p = [1/(1+np.exp(-i)) for i in z]
    return p


def p_matrix(data, model, n):
    p = np.zeros((n, n))
    k = 0
    for i in range(n * n):
        if i // n != n -1 and i % n != 0 and i % n != n - 1 and i // n != i % n:
            p[i//n, i % n] = obtain_probabilities(data[k].reshape((1, -1)), model)[0]
            k += 1
    return p


def predictions_matrix(data, model, n):
    p = np.full((n, n), "neg")
    k = 0
    for i in range(n * n):
        if i // n != n - 1 and i % n != 0 and i % n != n - 1 and i // n != i % n:
            p[i//n, i % n] = model.predict(data[k].reshape((1, -1)))[0]
            k += 1
    print(k, len(data))
    return p

