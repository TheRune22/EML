# Imports

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal as gaussian

data = pd.read_csv("faithful.csv", delimiter="\t", index_col=0)
x_min = 1
x_max = 6
y_min = 40
y_max = 100

# Plotting Data

fig, ax = plt.subplots()

ax.scatter(data["eruptions"], data["waiting"])
ax.set_xlabel("eruptions [minutes]")
ax.set_ylabel("waiting [minutes]")
ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)

plt.show()

# EM

X = data.to_numpy()


def EM(X, K):
    N, D = X.shape

    # Initialize
    means = X[np.random.choice(N, K, replace=False)]

    covariances = np.full((K, D, D), np.cov(X, rowvar=False))

    mixing_coefficients = np.full(K, 1 / K)

    # Precompute all \pi_k * N(x_n | \mu_k, \Sigma_k) terms
    pi_N = np.array([
        np.array([
            mixing_coefficients[k] * gaussian.pdf(X[n], mean=means[k], cov=covariances[k])
            for k in range(K)])
        for n in range(N)
    ])

    log_likelihood = np.sum(np.log(np.sum(pi_N, 1)))

    iterating = True

    while iterating:

        # E step

        responsibilities = pi_N / np.sum(pi_N, 1, keepdims=True)

        # M step

        N_ks = np.sum(responsibilities, axis=0, keepdims=True)

        means = 1/N_ks.T * (responsibilities.T @ X)

        covariances = np.array([1/N_ks[:, k] * np.sum(np.array([responsibilities[n, k] * (X[n] - means[k]).reshape(-1, 1) @ (X[n] - means[k]).reshape(1, -1) for n in range(N)]), axis=0) for k in range(K)])

        mixing_coefficients = (N_ks / N).reshape(-1,)

        # Precompute all \pi_k * N(x_n | \mu_k, \Sigma_k) terms
        pi_N = np.array([
            np.array([
                mixing_coefficients[k] * gaussian.pdf(X[n], mean=means[k], cov=covariances[k])
                for k in range(K)])
            for n in range(N)
        ])

        # Evaluate likelihood

        new_log_likelihood = np.sum(np.log(np.sum(pi_N, 1)))

        if new_log_likelihood - log_likelihood < 1e-6:
            iterating = False

        log_likelihood = new_log_likelihood

    return mixing_coefficients, means, covariances, pi_N / np.sum(pi_N, 1, keepdims=True), log_likelihood


mixing_coefficients, means, covariances, responsibilities, log_likelihood = EM(X, 2)

print(f"Final log likelihood: {log_likelihood}")

# Plot

# Setup grid
num = 1000
xs = np.linspace(x_min, x_max, num)
ys = np.linspace(y_min, y_max, num)
XX, YY = np.meshgrid(xs, ys)
xys = np.dstack((XX, YY))


def mixture_pdf(x, mixing_coefficients, means, covariances):
    # Calculate value of PDF at each point in grid
    K = len(mixing_coefficients)
    return np.sum(np.array([mixing_coefficients[k] * gaussian.pdf(x, mean=means[k], cov=covariances[k]) for k in range(K)]), axis=0)


def color_2responsibilities(x, mixing_coefficients, means, covariances):
    # Calculate responsibility at each point in grid
    K = len(mixing_coefficients)
    pi_N = np.array([
        np.array([
            mixing_coefficients[k] * gaussian.pdf(x[i], mean=means[k], cov=covariances[k])
            for k in range(K)])
        for i in range(x.shape[0])
    ])
    responsibilities = pi_N / np.sum(pi_N, 1, keepdims=True)

    # Move axis to simplify concatenation
    responsibilities = np.moveaxis(responsibilities, 1, 0)
    responsibilities = np.concatenate((responsibilities, np.full((1,) + x.shape[:-1], 0)))

    # Move axis to enable use as color
    responsibilities = np.moveaxis(responsibilities, 0, -1)

    return responsibilities


def plot_pdf(xs, ys, xys, mixing_coefficients, means, covariances):
    fig, ax = plt.subplots()
    zs = mixture_pdf(xys, mixing_coefficients, means, covariances)
    ax.contourf(xs, ys, zs)
    ax.scatter(data["eruptions"], data["waiting"], edgecolors='black', s=20, alpha=0.8)
    ax.scatter(means[:, 0], means[:, 1], edgecolors='black', facecolors='yellow', marker='X', s=100, c='yellow')
    ax.set_xlabel("eruptions [minutes]")
    ax.set_ylabel("waiting [minutes]")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    plt.show()


# Plot PDF
plot_pdf(xs, ys, xys, mixing_coefficients, means, covariances)

# Plot responsibility
fig, ax = plt.subplots()

responsibility_colors = np.concatenate((responsibilities, np.full((responsibilities.shape[0], 1), 0)), 1)
responsibility_img = color_2responsibilities(xys, mixing_coefficients, means, covariances)

ax.imshow(responsibility_img, extent=(x_min, x_max, y_min, y_max), aspect='auto', origin='lower')
ax.scatter(data["eruptions"], data["waiting"], c=responsibility_colors, edgecolors='black')
ax.scatter(means[:, 0], means[:, 1], edgecolors='black', facecolors='yellow', marker='X', s=100, c='yellow')

ax.set_xlabel("eruptions [minutes]")
ax.set_ylabel("waiting [minutes]")
ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)

plt.show()

#%% K = 3

mixing_coefficients, means, covariances, responsibilities, log_likelihood = EM(X, 3)

print(f"Final log likelihood: {log_likelihood}")

# Plot PDF
plot_pdf(xs, ys, xys, mixing_coefficients, means, covariances)


# Plot responsibility
def color_3responsibilities(x, mixing_coefficients, means, covariances):
    # Calculate responsibility at each point in grid
    K = len(mixing_coefficients)
    pi_N = np.array([
        np.array([
            mixing_coefficients[k] * gaussian.pdf(x[i], mean=means[k], cov=covariances[k])
            for k in range(K)])
        for i in range(x.shape[0])
    ])
    responsibilities = pi_N / np.sum(pi_N, 1, keepdims=True)

    # Move axis to enable use as color
    responsibilities = np.moveaxis(responsibilities, 1, -1)

    return responsibilities

fig, ax = plt.subplots()

responsibility_img = color_3responsibilities(xys, mixing_coefficients, means, covariances)

ax.imshow(responsibility_img, extent=(x_min, x_max, y_min, y_max), aspect='auto', origin='lower')
ax.scatter(data["eruptions"], data["waiting"], c=responsibilities, edgecolors='black')
ax.scatter(means[:, 0], means[:, 1], edgecolors='black', facecolors='yellow', marker='X', s=100, c='yellow')

ax.set_xlabel("eruptions [minutes]")
ax.set_ylabel("waiting [minutes]")
ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)

plt.show()


#%% K = 6

mixing_coefficients, means, covariances, responsibilities, log_likelihood = EM(X, 6)

print(f"Final log likelihood: {log_likelihood}")

# Plot PDF
plot_pdf(xs, ys, xys, mixing_coefficients, means, covariances)

# Plot responsibility
colors = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0.5, 0.5, 0], [0, 0.5, 0.5], [0.5, 0, 0.5]])


def color_6responsibilities(x, mixing_coefficients, means, covariances):
    # Calculate responsibility at each point in grid
    K = len(mixing_coefficients)
    pi_N = np.array([
        np.array([
            mixing_coefficients[k] * gaussian.pdf(x[i], mean=means[k], cov=covariances[k])
            for k in range(K)])
        for i in range(x.shape[0])
    ])
    responsibilities = pi_N / np.sum(pi_N, 1, keepdims=True)

    # Move axis to simplify concatenation
    responsibilities = np.moveaxis(responsibilities, 1, 0)

    responsibilities = colors.reshape((3, 6, 1, 1)) * responsibilities[np.newaxis, :]

    responsibilities = np.sum(responsibilities, 1)

    # Move axis to enable use as color
    responsibilities = np.moveaxis(responsibilities, 0, -1)

    return responsibilities


fig, ax = plt.subplots()

responsibility_colors = colors.reshape((3, 1, 6)) * responsibilities[np.newaxis, :]
responsibility_colors = np.sum(responsibility_colors, 2)
responsibility_colors = np.moveaxis(responsibility_colors, 0, -1)

responsibility_img = color_6responsibilities(xys, mixing_coefficients, means, covariances)

ax.imshow(responsibility_img, extent=(x_min, x_max, y_min, y_max), aspect='auto', origin='lower')
ax.scatter(data["eruptions"], data["waiting"], c=responsibility_colors, edgecolors='black')
ax.scatter(means[:, 0], means[:, 1], edgecolors='black', facecolors='yellow', marker='X', s=100, c='yellow')

ax.set_xlabel("eruptions [minutes]")
ax.set_ylabel("waiting [minutes]")
ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)

plt.show()


#%% K = 10

mixing_coefficients, means, covariances, responsibilities, log_likelihood = EM(X, 10)

print(f"Final log likelihood: {log_likelihood}")

# Plot PDF
plot_pdf(xs, ys, xys, mixing_coefficients, means, covariances)
