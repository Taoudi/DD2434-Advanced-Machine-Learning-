import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from mpl_toolkits.mplot3d import Axes3D

features = 1
output = 1


def prior_plot():
    # Parameters to set
    mu_x = 0
    variance_x = 1

    mu_y = 0
    variance_y = 1

    # Create grid and multivariate normal
    x = np.linspace(-2, 2, 2000)
    y = np.linspace(-2, 2, 2000)
    X, Y = np.meshgrid(x, y)
    pos = np.empty(X.shape + (2,))
    pos[:, :, 0] = X
    pos[:, :, 1] = Y
    rv = stats.multivariate_normal([mu_x, mu_y], [[variance_x, 0], [0, variance_y]])

    # Make a 3D plot
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_surface(X, Y, rv.pdf(pos), cmap='viridis', linewidth=0)
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    plt.show()


def parameters():
    n = 6
    w0 = 0.5
    w1 = -1.5
    x_row = np.arange(-1, 1+(2/n), (2/n))
    print(x_row)
    t_row = w0 * x_row + w1 + np.random.normal(0, 0.8)
    t_col = t_row.reshape(-1, 1)

    x_col = np.c_[np.ones(n+1), x_row]
    x_row = np.transpose(x_col)

    tau = 1
    sigma = 1
    # sigma = np.cov(x)

    inverse = np.linalg.inv(x_row @ x_col + (1 / tau) * np.identity(2))
    mu = (1 / sigma) * inverse @ x_row @ t_col
    return mu, inverse


def plot(mu, sig):
    # Create grid and multivariate normal
    x = np.linspace(-4, 4, 2000)
    y = np.linspace(-4, 4, 2000)
    X, Y = np.meshgrid(x, y)
    pos = np.empty(X.shape + (2,))
    pos[:, :, 0] = X;
    pos[:, :, 1] = Y
    rv = stats.multivariate_normal(mu.ravel(), sig)

    # Make a 3D plot
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_surface(X, Y, rv.pdf(pos), cmap='viridis', linewidth=0)
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    plt.show()


def sampling(mu, co):
    times = 5
    i = 0
    while i < times:
        f = np.random.multivariate_normal(mu.ravel(), co)
        if i == 0:
            arr = f
        else:
            arr = np.c_[arr, np.array(f)]
        i += 1
    lines = np.transpose(arr)
    print(lines)
    for line in lines:
        plt.plot(np.arange(10), line[1] * np.arange(10) + line[0])
    axes = plt.gca()
    axes.set_xlim([0, 9])
    axes.set_ylim([-5, 8])
    plt.show()


mu, sig = parameters()
#prior_plot()
#sampling(mu, sig)
plot(mu, sig)
