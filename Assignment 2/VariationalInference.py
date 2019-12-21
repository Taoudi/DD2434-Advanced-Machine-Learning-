import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from mpl_toolkits.mplot3d import Axes3D

N = 10
true_mean = 0
true_precision = 1


# generate random dataset X
def data_set():
    return np.random.normal(true_mean, np.sqrt(true_precision ** (-1)), N)


# 10.21 ( Generate dataset Y/D)
def likelihood(data):
    N = len(data)
    sum_data = (data - true_mean) ** 2
    sum_data = np.sum(sum_data)
    D = (true_precision / (2 * np.pi)) ** (N / 2) * np.exp(-(true_precision / 2) * sum_data)
    return D


# 10.26, 10.27
def gaussian_parameters(lambda0, mu0, a0, b0, X):
    N = len(X)
    mu_n = (lambda0 * mu0 + N * np.average(X)) / (lambda0 + N)
    if b0 == 0 or a0 == 0:
        expected_tau = 1
    else:
        expected_tau = (a0 / b0)
    lambda_n = (lambda0 + N) * expected_tau
    return mu_n, lambda_n


# E[mu**2], Needed to calculate b_n
def expected_mu(lamb0, X, mu0, mu_n, lamb_n):
    E_mu2 = lamb_n ** (-1) + mu_n ** 2
    square_sum = np.sum(X ** 2 - 2 * X * mu_n + E_mu2)
    return square_sum + lamb0 * (mu0 ** 2 - 2 * mu0 * mu_n + E_mu2)


def gamma_parameters(a0, b0, lambda0, mu0, lamb_n, mu_n, X):
    N = len(X)
    a_n = a0 + (N + 1) / 2
    b_n = b0 + (1 / 2) * expected_mu(lambda0, X, mu0, mu_n, lamb_n)
    return a_n, b_n


# Repeated (hopefully until convergence)
def VI(a0, b0, mu0, lamb0, X):
    times = 10
    i = 0
    while True:
        mu, lamb = gaussian_parameters(lamb0, mu0, a0, b0, X)
        a, b = gamma_parameters(a0, b0, lamb0, mu0, lamb, mu, X)
        mu0 = mu
        lamb0 = lamb
        a0 = a
        b0 = b
        i += 1
        if i == times:
            return mu, lamb, a, b


# Once
def VariationalInference(a0, b0, mu0, lamb0, X):
    mu, lamb = gaussian_parameters(lamb0, mu0, a0, b0, X)
    a, b = gamma_parameters(a0, b0, lamb0, mu0, lamb, mu, X)
    mu, lamb = gaussian_parameters(lamb0, mu, a, b, X)
    return mu, lamb, a, b


"""def plot(mean, variance, a, b):
    x_axis = np.arange(-10, 10, 0.001)
    # Mean = 0, SD = 2.
    plt.plot(stats.gamma.pdf(x_axis, a, scale=1 / b), stats.norm.pdf(x_axis, mean, variance ** (-1)))
    plt.show()"""


def q_mu(x):
    return stats.norm.pdf(x, mu, np.sqrt(1 / lam))


def q_tau(tau):
    return stats.gamma.pdf(tau, a, loc=0, scale=1 / b)


def plot():
    mus = np.linspace(-4, 4, 100)
    taus = np.linspace(-3, 3, 100)
    M, T = np.meshgrid(mus, taus, indexing="ij")
    Z = np.zeros_like(M)

    for i in range(Z.shape[0]):
        for j in range(Z.shape[1]):
            Z[i][j] = q_mu(mus[i]) * q_tau(taus[j])

    plt.contour(M, T, Z)
    plt.show()


mu, lam, a, b = VariationalInference(0, 0, 0, 0, data_set())

plot()
