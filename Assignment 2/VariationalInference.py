import numpy as np


def likelihood(X, tau, mu):
    N = len(X)

    x = np.arange(-100, 100, 1)
    sum_data = (x - mu) ** 2
    D = np.exp((tau / (2 * np.pi)) ** (N / 2), -(tau / 2) * sum_data)
    return D


def expected_tau():
    wtf = 0
    return wtf


def gaussian_parameters(lambda0, mu0, X):
    N = len(X)
    mu_n = (lambda0 * mu0 + N * np.average(X)) / (lambda0 + N)
    lambda_n = (lambda0 + N) * expected_tau()
    return mu_n, lambda_n


def expected_mu():
    wtf = 0
    return wtf


def gamma_parameters(a0, b0, N):
    a_n = a0 + N
    b_n = b0 + (1 / 2) * expected_mu()
    return a_n, b_n
