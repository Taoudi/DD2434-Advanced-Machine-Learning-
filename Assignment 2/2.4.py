import numpy as np


def likelihood(X, tau, mu):
    N = len(X)

    x = np.arange(-100, 100, 1)
    sum_data = (x - mu) ** 2
    D = np.exp((tau / (2 * np.pi)) ** (N / 2), -(tau / 2) * sum_data)
    return D



