import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from mpl_toolkits.mplot3d import Axes3D
tau = 10

def kernel(x, y):
    tau = 1
    var = 1
    return var * np.exp(-(np.transpose(x - y) @ (x - y)) / np.power(tau, 2))


def cov(x):

    variance = 1
    i = 0
    arr = 0
    for var in x:
        temp = variance * np.exp(-((var - x) ** 2) / (tau ** 2))
        if i == 0:
            arr = temp
        else:
            arr = np.c_[arr, temp]
        i += 1
    return arr


def gauss_it(cov):
    times = 10
    mu = np.zeros(len(cov))
    i = 0
    arr = 0
    while i < times:
        f = np.random.multivariate_normal(mu, cov)
        print(f)
        if i == 0:
            arr = f
        else:
            arr = np.c_[arr, np.array(f)]
        i += 1
    lines = np.transpose(arr)
    for line in lines:
        #print(line)
        plt.plot(np.arange(0, len(cov), 1), line)
    plt.xlabel("x")
    plt.ylabel("f")
    plt.title("Gaussian Prior for l = " + str(tau))
    plt.show()


def data():
    xt = np.array([-4, -3, -2, -1, 0, 2, 3, 5])
    x = xt.reshape(-1, 1)
    t = (2 + pow((0.5 * x - 1), 2) * np.sin(3 * x) + np.random.normal(0, 3))
    return x, t


def parameters():
    n = 200
    w0 = 0.5
    w1 = -1.5
    x_col = np.arange(-1, 1 + (2 / n), (2 / n)).reshape(-1, 1)
    return x_col


x, t = data()
x_col = parameters()

# print(np.mean(t))
gauss_it(cov(x_col))
