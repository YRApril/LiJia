import numpy as np
import matplotlib.pyplot as plt

name = "YOUR NAME HERE"
print("Hello {0}!".format(name))

# %matplotlib inline
from matplotlib import rcParams

rcParams["savefig.dpi"] = 100  # This makes all the plots a little bigger.

# Load the data from the CSV file.
x, y, yerr = np.loadtxt("linear.csv", delimiter=",", unpack=True)

# Plot the data with error bars.
plt.errorbar(x, y, yerr=yerr, fmt=".k", capsize=0)
plt.xlim(0, 5);

A = np.vander(x, 2)  # Take a look at the documentation to see what this function does!
ATA = np.dot(A.T, A / yerr[:, None] ** 2)
w = np.linalg.solve(ATA, np.dot(A.T, y / yerr ** 2))
V = np.linalg.inv(ATA)

plt.errorbar(x, y, yerr=yerr, fmt=".k", capsize=0)
for m, b in np.random.multivariate_normal(w, V, size=50):
    plt.plot(x, m * x + b, "g", alpha=0.1)
plt.xlim(0, 5);


def lnlike_linear((m, b)):
    raise NotImplementedError("Delete this placeholder and implement this function")


p_1, p_2 = (0.0, 0.0), (0.01, 0.01)
ll_1, ll_2 = lnlike_linear(p_1), lnlike_linear(p_2)
if not np.allclose(ll_2 - ll_1, 535.8707738280209):
    raise ValueError("It looks like your implementation is wrong!")
print("☺︎")


def lnprior_linear((m, b)):
    if not (-10 < m < 10):
        return -np.inf
    if not (-10 < b < 10):
        return -np.inf
    return 0.0


def lnpost_linear(theta):
    return lnprior_linear(theta) + lnlike_linear(theta)


def metropolis_step(lnpost_function, theta_t, lnpost_t, step_cov):
    raise NotImplementedError("Delete this placeholder and implement this function")


lptest = lambda x: -0.5 * np.sum(x ** 2)
th = np.array([0.0])
lp = 0.0
chain = np.array([th for th, lp in (metropolis_step(lptest, th, lp, [[0.3]])
                                    for _ in range(10000))])
if np.abs(np.mean(chain)) > 0.1 or np.abs(np.std(chain) - 1.0) > 0.1:
    raise ValueError("It looks like your implementation is wrong!")
print("☺︎")

# Edit these guesses.
m_initial = 2.
b_initial = 0.45

# You shouldn't need to change this plotting code.
plt.errorbar(x, y, yerr=yerr, fmt=".k", capsize=0)
for m, b in np.random.multivariate_normal(w, V, size=24):
    plt.plot(x, m_initial * x + b_initial, "g", alpha=0.1)
plt.xlim(0, 5)

# Edit this line to specify the proposal covariance:
step = np.diag([1e-6, 1e-6])

# Edit this line to choose the number of steps you want to take:
nstep = 50000

# Edit this line to set the number steps to discard as burn-in.
nburn = 1000

# You shouldn't need to change any of the lines below here.
p0 = np.array([m_initial, b_initial])
lp0 = lnpost_linear(p0)
chain = np.empty((nstep, len(p0)))
for i in range(len(chain)):
    p0, lp0 = metropolis_step(lnpost_linear, p0, lp0, step)
    chain[i] = p0

# Compute the acceptance fraction.
acc = float(np.any(np.diff(chain, axis=0), axis=1).sum()) / (len(chain) - 1)
print("The acceptance fraction was: {0:.3f}".format(acc))

# Plot the traces.
fig, axes = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
axes[0].plot(chain[:, 0], "k")
axes[0].axhline(w[0], color="g", lw=1.5)
axes[0].set_ylabel("m")
axes[0].axvline(nburn, color="g", alpha=0.5, lw=2)
axes[1].plot(chain[:, 1], "k")
axes[1].axhline(w[1], color="g", lw=1.5)
axes[1].set_ylabel("b")
axes[1].axvline(nburn, color="g", alpha=0.5, lw=2)
axes[1].set_xlabel("step number")
axes[0].set_title("acceptance: {0:.3f}".format(acc))

if np.any(np.abs(np.mean(chain, axis=0) - w) > 0.01) or np.any(np.abs(np.cov(chain, rowvar=0) - V) > 1e-4):
    raise ValueError("It looks like your implementation is wrong!")
print("☺︎")

import triangle

triangle.corner(chain[nburn:, :], labels=["m", "b"], truths=w);

plt.errorbar(x, y, yerr=yerr, fmt=".k", capsize=0)
for m, b in chain[nburn + np.random.randint(len(chain) - nburn, size=50)]:
    plt.plot(x, m * x + b, "g", alpha=0.1)
plt.xlim(0, 5);

# Edit these guesses.
alpha_initial = 100
beta_initial = -1

# These are the edges of the distribution (don't change this).
a, b = 1.0, 5.0

# Load the data.
events = np.loadtxt("poisson.csv")

# Make a correctly normalized histogram of the samples.
bins = np.linspace(a, b, 12)
weights = 1.0 / (bins[1] - bins[0]) + np.zeros(len(events))
plt.hist(events, bins, range=(a, b), histtype="step", color="k", lw=2, weights=weights)

# Plot the guess at the rate.
xx = np.linspace(a, b, 500)
plt.plot(xx, alpha_initial * xx ** beta_initial, "g", lw=2)

# Format the figure.
plt.ylabel("number")
plt.xlabel("x");


def lnlike_poisson((alpha, beta)):
    raise NotImplementedError("Delete this placeholder and implement this function")


p_1, p_2 = (1000.0, -1.), (1500., -2.)
ll_1, ll_2 = lnlike_poisson(p_1), lnlike_poisson(p_2)
if not np.allclose(ll_2 - ll_1, 337.039175916):
    raise ValueError("It looks like your implementation is wrong!")
print("☺︎")


def lnprior_poisson((alpha, beta)):
    if not (0 < alpha < 1000):
        return -np.inf
    if not (-10 < beta < 10):
        return -np.inf
    return 0.0


def lnpost_poisson(theta):
    return lnprior_poisson(theta) + lnlike_poisson(theta)

# Edit this line to specify the proposal covariance:
step = np.diag([1000., 4.])

# Edit this line to choose the number of steps you want to take:
nstep = 50000

# Edit this line to set the number steps to discard as burn-in.
nburn = 1000

# You shouldn't need to change any of the lines below here.
p0 = np.array([alpha_initial, beta_initial])
lp0 = lnpost_poisson(p0)
chain = np.empty((nstep, len(p0)))
for i in range(len(chain)):
    p0, lp0 = metropolis_step(lnpost_poisson, p0, lp0, step)
    chain[i] = p0

# Compute the acceptance fraction.
acc = float(np.any(np.diff(chain, axis=0), axis=1).sum()) / (len(chain)-1)
print("The acceptance fraction was: {0:.3f}".format(acc))

# Plot the traces.
fig, axes = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
axes[0].plot(chain[:, 0], "k")
axes[0].set_ylabel("alpha")
axes[0].axvline(nburn, color="g", alpha=0.5, lw=2)
axes[1].plot(chain[:, 1], "k")
axes[1].set_ylabel("beta")
axes[1].axvline(nburn, color="g", alpha=0.5, lw=2)
axes[1].set_xlabel("step number")
axes[0].set_title("acceptance: {0:.3f}".format(acc));


triangle.corner(chain[nburn:], labels=["alpha", "beta"], truths=[500, -2]);


plt.hist(events, bins, range=(a, b), histtype="step", color="k", lw=2, weights=weights)

# Plot the guess at the rate.
xx = np.linspace(a, b, 500)
for alpha, beta in chain[nburn+np.random.randint(len(chain)-nburn, size=50)]:
    plt.plot(xx, alpha * xx ** beta, "g", alpha=0.1)

# Format the figure.
plt.ylabel("number")
plt.xlabel("x");