import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D

N = 10
true_mu, true_lam, true_a, true_b = 3, 1, 2, 6

def gauss_sample(mean, prec):
    return np.random.normal(mean, np.sqrt(prec ** -1))


def gamma_sample(al, be):
    return np.random.gamma(al, 1 / be)


true_mean = true_mu
true_precision = true_a / true_b


# generate random dataset X
def data_set(m, p):
    return np.random.normal(m, np.sqrt(p ** (-1)), N)


# 10.21 ( Generate dataset Y/D)
def likelihood(data, par1, par2):
    N = len(data)
    sum_data = (data - par1) ** 2
    sum_data = np.sum(sum_data)
    D = (par2 / (2 * np.pi)) ** (N / 2) * np.exp(-(par2 / 2) * sum_data)
    return D


# E[mu**2], Needed to calculate b_n
def expected_mu(lamb0, X, mu0, mu_n, lamb_n):
    E_mu2 = lamb_n ** (-1) + mu_n ** 2
    square_sum = np.sum((X ** 2) - (2 * X * mu_n) + E_mu2)
    return (1 / 2 * square_sum) + lamb0 * ((mu0 ** 2) - (2 * mu0 * mu_n) + E_mu2)


def approx_a(a0):
    return a0 + ((N + 1) / 2)


def approx_mu(l0, m0, X):
    return (l0 * m0 + N * np.average(X)) / (l0 + N)


def approx_lambda(l0, a_n, b_n):
    return (l0 + N) * (a_n / b_n)


def approx_b(m0, m_n, l_n, l0, b0):
    return b0 + expected_mu(l0, X, m0, m_n, l_n)


# Once
def VariationalInference(mu0, lamb0, a0, b0, X, iterations):
    i = 0
    la = lamb0
    be = b0
    mu = mu0
    al = a0
    while i < iterations:
        a0 = al
        mu0 = mu
        lamb0 = la
        b0 = be
        al = approx_a(a0)
        mu = approx_mu(lamb0, mu0, X)
        la = approx_lambda(lamb0, al, be)
        be = approx_b(mu0, mu, la, lamb0, b0)
        print(al)
        i += 1
        if i == iterations:
            return mu, la, al, be


"""def plot(mean, variance, a, b):
    x_axis = np.arange(-10, 10, 0.001)
    # Mean = 0, SD = 2.
    plt.plot(stats.gamma.pdf(x_axis, a, scale=1 / b), stats.norm.pdf(x_axis, mean, variance ** (-1)))
    plt.show()"""


def q_mu(x, mean, precision):
    return stats.norm.pdf(x, mean, np.sqrt(1 / precision))


def q_tau(tau, alpha, beta):
    return stats.gamma.pdf(tau, alpha, loc=0, scale=(1 / beta))


def true_plot(mean, precision, alpha, beta, data):
    print("Generated mean, tau: " + str(true_mean) + " ," + str(true_precision))
    print("Actual mean, lambda , alpha ,beta: " + str(mean) + " ," + str(precision) + " ," + str(alpha) + " ," + str(
        beta))
    mus = np.linspace(true_mean - 5, true_mean + 5, 100)
    taus = np.linspace(true_precision - 0.5, true_precision + 0.5, 100)
    M, T = np.meshgrid(mus, taus, indexing="ij")
    Z = np.zeros_like(M)
    for i in range(Z.shape[0]):
        for j in range(Z.shape[1]):
            Z[i][j] = q_mu(mus[i], true_mean, true_precision * true_lam) * q_tau(taus[j], alpha, beta) * likelihood(data
                                                                                                                    ,
                                                                                                                    mus[
                                                                                                                        i],
                                                                                                                    taus[
                                                                                                                        j])
    plt.contour(M, T, Z, colors='green')


def plot(mean, precision, alpha, beta):
    print("Observed mean, tau: " + str(mean) + " ," + str(alpha / beta))
    print(
        "Observed mean, tau, lambda , alpha ,beta: " + str(mean) + " ," + str(precision) + " ," + str(
            alpha) + " ," + str(
            beta))
    mus = np.linspace(true_mean - 5, true_mean + 5, 100)
    taus = np.linspace(true_precision - 0.5, true_precision + 0.5, 100)
    M, T = np.meshgrid(mus, taus, indexing="ij")
    Z = np.zeros_like(M)

    for i in range(Z.shape[0]):
        for j in range(Z.shape[1]):
            Z[i][j] = q_mu(mus[i], mean, precision) * q_tau(taus[j], alpha, beta)
    plt.contour(M, T, Z, colors='red')
    plt.scatter(true_mean, true_precision, color="black")


# X = data_set(true_mean, true_precision)
X = np.array([4.91683995, 2.28125975, 0.95866215, 1.34756688, 0.88799854, 2.22666351,
              3.93950624, 4.88324244, 0.20460881, 2.40607224])
iter = 3
m, l, a, b = 10, 10, 50, 100

m, l, a, b = VariationalInference(m, l, a, b, X, iter)

custom_lines = [Line2D([0], [0], color="red", lw=4),
                Line2D([0], [0], color="green", lw=4)]

fig, ax = plt.subplots()
ax.legend(custom_lines, ['Inferred', 'True'])
plot(m, l, a, b)
true_plot(true_mu, true_lam, true_a, true_b, X)
plt.xlabel("mean")
plt.ylabel("precision")
plt.title("True Posterior and Inferred Posterior, Iterations =" + str(iter))

plt.show()
