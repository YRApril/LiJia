import numpy as np
import scipy
import scipy.stats
import matplotlib as mpl
import math
import matplotlib.pyplot as plt
# Part 3: A real world example

from MCMC import metropolis_hastings, acceptance

sunspots = np.loadtxt("data/SN_m_tot_V2.0.csv", delimiter=';')

# years is the third column
years = sunspots[:, 2]  # 年

# activity is the 4th column
activity = sunspots[:, 3] + 0.1  # 活动

print(years.shape, activity.shape)

fig = plt.figure(figsize=(20, 20))
ax = fig.add_subplot(2, 1, 1)

ax.plot(years, activity, label="Sunspot count")
ax.set_yticks(np.arange(0, 500, 100))
ax.set_xlabel("Years")
ax.set_ylabel("Monthly mean count of sunspots")
ax.set_title("Figure 7: Sunspots, 1749-2018")
ax.legend()
# fig.show()
# exit()

ax = fig.add_subplot(2, 1, 2)
print(years[432])
ax.plot(years[:432], activity[:432], label="Sunspot count")
ax.set_yticks(np.arange(0, 500, 100))
ax.set_xlabel("Years")
ax.set_ylabel("Monthly mean count of sunspots")
ax.set_title("Figure 8: Sunspots, 1749-1785")
ax.legend()
# fig.show()
# exit()

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(1, 1, 1)
ax.hist(activity, bins=40, density=True)
ax.set_xlabel("Sunspot count")
ax.set_ylabel("Frequency")
ax.set_title("Figure 9: Histogram showing the frequency of sunspot counts over 270 years (1749-2018)")
# fig.show()
# exit()


transition_model = lambda x: np.random.normal(x, [0.05, 5], (2,))


def prior(w):
    if (w[0] <= 0 or w[1] <= 0):
        return 0
    else:
        return 1


def manual_log_lik_gamma(x, data):
    return np.sum(
        (x[0] - 1) * np.log(data) - (1 / x[1]) * data - x[0] * np.log(x[1]) - np.log(math.gamma(x[0])))


def log_lik_gamma(x, data):
    return np.sum(np.log(scipy.stats.gamma(a=x[0], scale=x[1], loc=0).pdf(data)))


accepted, rejected = metropolis_hastings(manual_log_lik_gamma, prior, transition_model, [4, 10], 50000,
                                         activity, acceptance)

print(accepted.shape)
print(accepted[-10:])

fig = plt.figure(figsize=(10, 20))
ax = fig.add_subplot(3, 1, 1)
ax.plot(accepted[:50, 0], accepted[:50, 1], label="Path")
ax.plot(accepted[:50, 0], accepted[:50, 1], 'b.', label='Accepted')
ax.plot(rejected[:50, 0], rejected[:50, 1], 'rx', label='Rejected')
ax.set_xlabel("a")
ax.set_ylabel("b")
ax.legend()
ax.set_title("Figure 10: MCMC sampling for $a$ and $b$ with Metropolis-Hastings. First 50 samples are shown.")
# fig.show()
# exit()

ax = fig.add_subplot(3, 1, 2)
ax.plot(accepted[:, 0], accepted[:, 1], label="Path")
ax.plot(accepted[:, 0], accepted[:, 1], 'b.', label='Accepted', alpha=0.3)
ax.plot(rejected[:, 0], rejected[:, 1], 'rx', label='Rejected', alpha=0.3)
ax.set_xlabel("a")
ax.set_ylabel("b")
ax.legend()
ax.set_title("Figure 11: MCMC sampling for $a$ and $b$ with Metropolis-Hastings. All samples are shown.")
# fig.show()
# exit()

to_show = 50
ax = fig.add_subplot(3, 1, 3)
ax.plot(accepted[-to_show:, 0], accepted[-to_show:, 1], label="Path")
ax.plot(accepted[-to_show:, 0], accepted[-to_show:, 1], 'b.', label='Accepted', alpha=0.5)
ax.plot(rejected[-to_show:, 0], rejected[-to_show:, 1], 'rx', label='Rejected', alpha=0.5)
ax.set_xlabel("a")
ax.set_ylabel("b")
ax.legend()
ax.set_title("Figure 12: MCMC sampling for $a$ and $b$ with Metropolis-Hastings. Last 50 samples are shown.")
# fig.show()
# exit()

show = int(-0.5 * accepted.shape[0])
hist_show = int(-0.50 * accepted.shape[0])

fig = plt.figure(figsize=(15, 7))
ax = fig.add_subplot(1, 2, 1)
ax.plot(accepted[show:, 0])
ax.set_title("Figure 13: Trace for $a$")
ax.set_xlabel("Iteration")
ax.set_ylabel("a")

ax = fig.add_subplot(1, 2, 2)
ax.hist(accepted[hist_show:, 0], bins=20, density=True)
ax.set_ylabel("Frequency (normed)")
ax.set_xlabel("a")
ax.set_title("Figure 14: Histogram of $a$")
fig.tight_layout()
# fig.show()
# exit()

fig = plt.figure(figsize=(15, 7))
ax = fig.add_subplot(1, 2, 1)
ax.plot(accepted[show:, 1])
ax.set_title("Figure 15: Trace for $b$")
ax.set_xlabel("Iteration")
ax.set_ylabel("b")

ax = fig.add_subplot(1, 2, 2)
ax.hist(accepted[hist_show:, 1], bins=20, density=True)
ax.set_ylabel("Frequency (normed)")
ax.set_xlabel("b")
ax.set_title("Figure 16: Histogram of $b$")
fig.tight_layout()
# fig.show()
# exit()

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(1, 1, 1)
xbins, ybins = np.linspace(0.8, 1.2, 30), np.linspace(75, 90, 30)
counts, xedges, yedges, im = ax.hist2d(accepted[hist_show:, 0], accepted[hist_show:, 1], density=True,
                                       bins=[xbins, ybins])
ax.set_xlabel("a")
ax.set_ylabel("b")
fig.colorbar(im, ax=ax)
ax.set_title("2D histogram showing the joint distribution of $a$ and $b$")
# fig.show()
# exit()

show = -int(0.5 * accepted.shape[0])

mu = accepted[show:, 0].mean()
sigma = accepted[show:, 1].mean()
print(mu, sigma)

model = lambda t, mu, sigma: np.random.gamma(mu, sigma, t)  # 方法

t = np.arange(activity.shape[0])
observation_gen = model(t.shape[0], mu, sigma)
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(1, 1, 1)

ax.hist(observation_gen, bins=np.linspace(0, 500, 50), density=True, label="Predicted values")
ax.hist(activity, bins=np.linspace(0, 500, 50), alpha=0.5, density=True, label="Original values")
ax.set_xlabel("Count")
ax.set_ylabel("Frequency")
ax.set_title("Figure 17: Posterior distribution of predicitons")
ax.legend()

mean_acc_0 = accepted[show:, 0].mean()
mean_acc_1 = accepted[show:, 1].mean()
print(mean_acc_0, mean_acc_1)

lag = np.arange(1, 100)


def autocorr(accepted, lag):
    num_0 = 0
    denom_0 = 0
    num_1 = 0
    denom_1 = 0
    for i in range(accepted.shape[0] - lag):
        num_0 += (accepted[i, 0] - mean_acc_0) * (accepted[i + lag, 0] - mean_acc_0)
        num_1 += (accepted[i, 1] - mean_acc_1) * (accepted[i + lag, 1] - mean_acc_1)
        denom_0 += (mean_acc_0 - accepted[i, 0]) ** 2
        denom_1 += (mean_acc_1 - accepted[i, 1]) ** 2
    rk_0 = num_0 / denom_0
    rk_1 = num_1 / denom_1
    return rk_0, rk_1


accepted_reversed = accepted[show:, :]
result = np.zeros((2, lag.shape[0]))
# print(lag)
for l in lag:
    result[:, l - 1] = autocorr(accepted_reversed, l)

###Instead of writing an autocorrelation function, one could simply use thee autocorr function provided in pymc3
# from pymc3.stats import autocorr

fig, ax = plt.subplots()
# ax.plot(lag, [autocorr(accepted[show:,1], l) for l in lags], label='auto b')
# ax.plot(lag, [autocorr(accepted[show:,0], l) for l in lags], label='auto a')
ax.plot(lag, result[1, :], label='Auto correlation for b')
ax.plot(lag, result[0, :], label='Auto correlation for a')
ax.legend(loc=0)
ax.set_title("Figure 18: Auto-correlation for a and b, for k=1..100")
ax.set(xlabel='lag', ylabel='autocorrelation', ylim=(-.1, 1))
