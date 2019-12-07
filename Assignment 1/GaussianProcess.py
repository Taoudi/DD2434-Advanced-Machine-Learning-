import math

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from mpl_toolkits.mplot3d import Axes3D

tau = 2
noise = math.sqrt(3)
inv_noise = 0


def kernel(x, y):
    tau = 1
    var = 1
    return var * np.exp(-(np.transpose(x - y) @ (x - y)) / np.power(tau, 2))


def cov(x, x2):
    variance = 1
    i = 0
    arr = 0
    for var in x:
        temp = variance * np.exp(-((var - x2) ** 2) / (tau ** 2))
        # print(temp)
        if i == 0:
            arr = temp
        else:
            arr = np.c_[arr, temp]
        i += 1
    return arr


def data():
    xt = np.array([-4, -3, -2, -1, 0, 2, 3, 5])
    x = xt.reshape(-1, 1)
    t = (2 + pow((0.5 * x - 1), 2) * np.sin(3 * x)) + np.random.normal(0, noise, x.shape)
    print(t)
    return x, t


def new_data():
    return np.arange(-10, 10, 0.05).reshape(-1, 1)


def gauss_it(mu, co):
    times = 5
    i = 0
    arr = 0
    while i < times:
        fff = np.random.multivariate_normal(mu.ravel(), co)
        if i == 0:
            arr = fff
        else:
            arr = np.c_[arr, np.array(fff)]
        i += 1
    dt, t = data()
    plt.scatter(dt, t, zorder=10)
    lines = np.transpose(arr)
    i = 2
    for line in lines:
        plt.plot(new_data(), line, zorder=i)
        i += 1
    plt.title("GP Posterior with l=" + str(tau))
    plt.xlabel("x")
    plt.ylabel("f")
    plt.show()


def gauss_it_mean(mu, co):
    times = 10
    i = 0
    arr = 0
    while i < times:
        fff = np.random.multivariate_normal(mu.ravel(), co)
        if i == 0:
            arr = fff
        else:
            arr = np.c_[arr, np.array(fff)]
        i += 1
    dt, t = data()
    plt.scatter(dt, t, zorder=10)
    lines = np.transpose(arr)
    meanline = lines.mean(axis=0)
    plt.plot(new_data(), meanline, zorder=10)
    for line in lines:
        plt.fill_between(new_data().ravel(), meanline, line, color='grey', alpha='1')
        i += 1
    plt.title("GP Posterior with l=" + str(tau) + "Noise Parameter = " + str(inv_noise))
    plt.xlabel("x")
    plt.ylabel("f")
    plt.show()


def sample_f(mu, cov):
    return np.random.multivariate_normal(mu, cov)


def parameters():
    n = 200
    w0 = 0.5
    w1 = -1.5
    x_col = np.arange(-1, 1 + (2 / n), (2 / n)).reshape(-1, 1)
    return x_col


def post_plot(mew, co):
    x_axis = np.arange(-6, 10, 1)
    # Mean = 0, SD = 2.
    plt.plot(x_axis, stats.norm.pdf(x_axis, mew, co))
    plt.show()


x, t = data()
x_new = new_data()

f = sample_f(np.zeros(len(x)), cov(x, x))
print(3 * np.identity(len(x)))

inverse = np.linalg.inv(cov(x, x) + (inv_noise ** 2) * np.identity(len(x)))

xnewx = np.transpose(cov(x_new, x))
mult = xnewx @ inverse
mu = mult @ t

var = cov(x_new, x_new) - mult @ np.transpose(cov(x, x_new))
gauss_it_mean(mu, var)
