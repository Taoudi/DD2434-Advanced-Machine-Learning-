import numpy as np

### Set Parameters
true_precision = 1
true_mean = 5


# 10.21 ( Generate dataset Y/D)
def likelihood(data):
    N = len(data)
    sum_data = (data - true_mean) ** 2
    sum = np.sum(sum_data)
    print(-(true_precision / 2) * sum)
    D = (true_precision / (2 * np.pi)) ** (N / 2) * np.exp(-(true_precision / 2) * sum)
    return D


# generate random dataset X
def data_set():
    N = 10
    lower = -10
    higher = 10
    x = np.arange(lower, higher + np.abs(higher - lower) / N, np.abs(higher - lower) / N)
    return x


print(likelihood(data_set()))


# 10.29 and 10.30
def expected_tau(a0, b0, lambda0, N):
    a = a0 + (N / 2)

    expected = 0  # do all calcs
    b = b0 + (1 / 2) * expected
    return a / b


# 10.26, 10.27
def gaussian_parameters(lambda0, mu0, a0, b0, X):
    N = len(X)
    mu_n = (lambda0 * mu0 + N * np.average(X)) / (lambda0 + N)
    lambda_n = (lambda0 + N) * expected_tau(a0, b0.lambda0, N)
    return mu_n, lambda_n


def expected_mu():
    wtf = 0
    return wtf


def gamma_parameters(a0, b0, N):
    a_n = a0 + N
    b_n = b0 + (1 / 2) * expected_mu()
    return a_n, b_n
