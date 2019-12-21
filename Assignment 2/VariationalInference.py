import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from mpl_toolkits.mplot3d import Axes3D

N = 100
true_mu, true_lam, true_a, true_b = 0, 0.1, 4, 2


def gauss_sample(mean, prec):
    return np.random.normal(mean, np.sqrt(prec ** -1))


def gamma_sample(al, be):
    return np.random.gamma(al, 1 / be)


true_mean = gauss_sample(true_mu, true_lam)
true_precision = gamma_sample(true_a, true_b)


# generate random dataset X
def data_set():
    return np.random.normal(true_mean, np.sqrt(true_precision ** (-1)), N)


# 10.21 ( Generate dataset Y/D)
def likelihood(data, par1, par2):
    N = len(data)
    sum_data = (data - par1) ** 2
    sum_data = np.sum(sum_data)
    D = (par2 / (2 * np.pi)) ** (N / 2) * np.exp(-(par2 / 2) * sum_data)
    return D


# 10.26, 10.27
def gaussian_parameters(lambda0, mu0, a0, b0, X):
    N = len(X)
    mu_n = (lambda0 * mu0 + np.sum(X)) / (lambda0 + N)
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

#def expected_mu2(lamb0, X, mu0, mu_n, lamb_n):


def gamma_parameters(a0, b0, lambda0, mu0, lamb_n, mu_n, X):
    N = len(X)
    a_n = a0 + (N + 1) / 2
    b_n = b0 + (1 / 2) * expected_mu(lambda0, X, mu0, mu_n, lamb_n)
    return a_n, b_n


# Repeated (hopefully until convergence)
def VI(a0, b0, mu0, lamb0, X):
    times = 1
    i = 0
    while True:
        mu, lamb = gaussian_parameters(lamb0, mu0, a0, b0, X)
        a, b = gamma_parameters(a0, b0, lamb0, mu0, lamb, mu, X)
        if i == 0:
            mu, lamb = gaussian_parameters(lamb0, mu0, a, b, X)
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
    mu0, lamb0 = lamb, mu
    # mu, lamb = gaussian_parameters(lamb0, mu0, a, b, X)
    # a, b = gamma_parameters(a, b, lamb0, mu0, lamb, mu, X)
    return mu, lamb, a, b


"""def plot(mean, variance, a, b):
    x_axis = np.arange(-10, 10, 0.001)
    # Mean = 0, SD = 2.
    plt.plot(stats.gamma.pdf(x_axis, a, scale=1 / b), stats.norm.pdf(x_axis, mean, variance ** (-1)))
    plt.show()"""


def q_mu(x, mean, precision):
    return stats.norm.pdf(x, mean, np.sqrt(1 / precision))


def q_tau(tau, alpha, beta):
    return stats.gamma.pdf(tau, alpha, loc=0, scale=1 / beta)


def true_plot(mean, precision, alpha, beta, data):
    print("Generated mean, tau: " + str(true_mean) + " ," + str(true_precision))
    print("Actual mean, lambda , alpha ,beta: " + str(mean) + " ," + str(precision) + " ," + str(alpha) + " ," + str(
        beta))
    mus = np.linspace(-10, 10, 100)
    taus = np.linspace(-0.1, 4, 100)
    M, T = np.meshgrid(mus, taus, indexing="ij")
    Z = np.zeros_like(M)
    for i in range(Z.shape[0]):
        for j in range(Z.shape[1]):
            Z[i][j] = q_mu(mus[i], mean, precision) * q_tau(taus[j], alpha, beta) * likelihood(data, true_mean,
                                                                                               true_precision)
    plt.contour(M, T, Z, colors='green')


def plot(mean, precision, alpha, beta):
    print(
        "Observed mean, tau, lambda , alpha ,beta: " + str(mean) + " ," + str(precision) + " ," + str(
            alpha) + " ," + str(
            beta))
    mus = np.linspace(true_mean - 2, true_mean + 2, 100)
    taus = np.linspace(true_precision - 2, true_precision + 2, 100)
    M, T = np.meshgrid(mus, taus, indexing="ij")
    Z = np.zeros_like(M)

    for i in range(Z.shape[0]):
        for j in range(Z.shape[1]):
            Z[i][j] = q_mu(mus[i], mean, precision) * q_tau(taus[j], alpha, beta)
    plt.contour(M, T, Z, colors='red')
    plt.scatter(true_mean, true_precision, color="black")


X = data_set()
i = 0
m, l, a, b = 14, 0.01, 4, 1
t = 5
while i < t:
    m, l, a, b = VariationalInference(m, l, a, b, X)
    i += 1

plot(m, l, a, b)
true_plot(true_mu, true_lam, true_a, true_b, X)
plt.xlabel("mean")
plt.ylabel("precision")

plt.show()

x = np.array([2, 3, 4])
