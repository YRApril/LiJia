# pandas and numpy for data manipulation
import pandas as pd
import numpy as np

# scipy for algorithms
import scipy
from scipy import stats

# pymc3 for Bayesian Inference, pymc built on t
import pymc3 as pm
import theano.tensor as tt
import scipy
from scipy import optimize

# matplotlib for plotting
import matplotlib.pyplot as plt
# %matplotlib inline
from IPython.core.pylabtools import figsize
import matplotlib

import json

s = json.load(open('../style/bmh_matplotlibrc.json'))
matplotlib.rcParams.update(s)
matplotlib.rcParams['figure.figsize'] = (10, 3)
matplotlib.rcParams['font.size'] = 14
matplotlib.rcParams['ytick.major.size'] = 20

# Number of samples for Markov Chain Monte Carlo
N_SAMPLES = 5000

# %%

# Data formatted in different notebook
sleep_data = pd.read_csv('data/sleep_data.csv')
wake_data = pd.read_csv('data/wake_data.csv')

# Labels for plotting
sleep_labels = ['9:00', '9:30', '10:00', '10:30', '11:00', '11:30', '12:00']
wake_labels = ['5:00', '5:30', '6:00', '6:30', '7:00', '7:30', '8:00']

print('Number of sleep observations %d' % len(sleep_data))

# %%

figsize(16, 6)

# Sleep data
plt.scatter(sleep_data['time_offset'], sleep_data['indicator'],
            s=60, alpha=0.01, facecolor='b', edgecolors='b')
plt.yticks([0, 1], ['Awake', 'Asleep'])
plt.xlabel('PM Time')
plt.title('Falling Asleep Data', size=18)
plt.xticks([-60, -30, 0, 30, 60, 90, 120], sleep_labels)

# Wake data
plt.scatter(wake_data['time_offset'], wake_data['indicator'],
            s=50, alpha=0.01, facecolor='r', edgecolors='r')
plt.yticks([0, 1], ['Awake', 'Asleep'])
plt.xlabel('AM Time')
plt.title('Waking Up Data')
plt.xticks([-60, -30, 0, 30, 60, 90, 120], wake_labels)

figsize(16, 6)


# Logistic function with only beta
def logistic(x, beta):
    return 1. / (1. + np.exp(beta * x))


# Plot examples with different betas
x = np.linspace(-5, 5, 1000)
for beta in [-5, -1, 0.5, 1, 5]:
    plt.plot(x, logistic(x, beta), label=r"$\beta$ = %.1f" % beta)

plt.legend()
plt.title(r'Logistic Function with Different $\beta$ values')

figsize(20, 8)


# Logistic function with both beta and alpha
def logistic(x, beta, alpha=0):
    return 1.0 / (1.0 + np.exp(np.dot(beta, x) + alpha))


x = np.linspace(-4, 6, 1000)

plt.plot(x, logistic(x, beta=1), label=r"$\beta = 1, \alpha = 0$", ls="--", lw=2)
plt.plot(x, logistic(x, beta=-1), label=r"$\beta = -1 \alpha = 0$", ls="--", lw=2)

plt.plot(x, logistic(x, -1, 1),
         label=r"$\beta = -1, \alpha = 1$", color="darkblue")
plt.plot(x, logistic(x, -1, -1),
         label=r"$\beta = -1, \alpha = -1$", color="skyblue")
plt.plot(x, logistic(x, -2, 5),
         label=r"$\beta = -2, \alpha = 5$", color="orangered")
plt.plot(x, logistic(x, -2, -5),
         label=r"$\beta = -2, \alpha = -5$", color="darkred")
plt.legend()
plt.ylabel('Probability')
plt.xlabel('t')
plt.title(r'Logistic Function with Varying $\beta$ and $\alpha$')

figsize(20, 8)
# Set up the plotting parameters
nor = stats.norm
x = np.linspace(-10, 10, 1000)
mu = (-5, 0, 4)
tau = (0.5, 1, 2.5)
colors = ("forestgreen", "navy", "darkred")

# Plot 3 pdfs for different normal distributions
params = zip(mu, tau, colors)
for param in params:
    y = nor.pdf(x, loc=param[0], scale=1 / param[1])
    plt.plot(x, y,
             label="$\mu = %d,\\\tau = %.1f$" % (param[0], param[1]),
             color=param[2])
    plt.fill_between(x, y, color=param[2], alpha=0.3)

plt.legend(prop={'size': 18})
plt.xlabel("$x$")
plt.ylabel("Probability Density", size=18)
plt.title("Normal Distributions", size=20)

# %matplotlib inline
import scipy.stats as stats
from IPython.core.pylabtools import figsize
import numpy as np

figsize(16, 8)

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

jet = plt.cm.jet
fig = plt.figure()
x = y = np.linspace(0, 5, 100)
X, Y = np.meshgrid(x, y)

plt.subplot(121)
norm_x = stats.norm.pdf(x, loc=0, scale=1)
norm_y = stats.norm.pdf(y, loc=0, scale=1)
M = np.dot(norm_x[:, None], norm_y[None, :])
im = plt.imshow(M, interpolation='none', origin='lower',
                cmap=jet)

plt.xlim(0, 40)
plt.ylim(0, 40)
plt.title("Parameter Search Space for Normal Priors.")

ax = fig.add_subplot(122, projection='3d')
ax.plot_surface(X, Y, M, cmap=plt.cm.jet)
ax.view_init(azim=390)
plt.title("Parameter Search Space for Normal Priors")

# Sort the values by time offset
sleep_data.sort_values('time_offset', inplace=True)

# Time is the time offset
time = np.array(sleep_data.loc[:, 'time_offset'])

# Observations are the indicator
sleep_obs = np.array(sleep_data.loc[:, 'indicator'])

# %%

with pm.Model() as sleep_model:
    # Create the alpha and beta parameters
    alpha = pm.Normal('alpha', mu=0.0, tau=0.01, testval=0.0)
    beta = pm.Normal('beta', mu=0.0, tau=0.01, testval=0.0)

    # Create the probability from the logistic function
    p = pm.Deterministic('p', 1. / (1. + tt.exp(beta * time + alpha)))

    # Create the bernoulli parameter which uses the observed dat
    observed = pm.Bernoulli('obs', p, observed=sleep_obs)

    # Starting values are found through Maximum A Posterior estimation
    # start = pm.find_MAP()

    # Using Metropolis Hastings Sampling
    step = pm.Metropolis()

    # Sample from the posterior using the sampling method
    # sleep_trace = pm.sample(N_SAMPLES, step=step, njobs=2)
    sleep_trace = pm.sample(N_SAMPLES, step=step, cores=1)

# Extract the alpha and beta samples
alpha_samples = sleep_trace["alpha"][5000:, None]
beta_samples = sleep_trace["beta"][5000:, None]

# %%

figsize(16, 10)

plt.subplot(211)
plt.title(r"""Distribution of $\alpha$ with %d samples""" % N_SAMPLES)

plt.hist(alpha_samples, histtype='stepfilled',
         color='darkred', bins=30, alpha=0.8, density=True)
plt.ylabel('Probability Density')

plt.subplot(212)
plt.title(r"""Distribution of $\beta$ with %d samples""" % N_SAMPLES)
plt.hist(beta_samples, histtype='stepfilled',
         color='darkblue', bins=30, alpha=0.8, density=True)
plt.ylabel('Probability Density')

# Time values for probability prediction
time_est = np.linspace(time.min() - 15, time.max() + 15, int(1e3))[:, None]

# Take most likely parameters to be mean values
alpha_est = alpha_samples.mean()
beta_est = beta_samples.mean()

# Probability at each time using mean values of alpha and beta
sleep_est = logistic(time_est, beta=beta_est, alpha=alpha_est)

# %%

figsize(16, 6)

plt.plot(time_est, sleep_est, color='navy',
         lw=3, label="Most Likely Logistic Model")
plt.scatter(time, sleep_obs, edgecolor='slateblue',
            s=50, alpha=0.2, label='obs')
plt.title('Probability Distribution for Sleep with %d Samples' % N_SAMPLES)
plt.legend(prop={'size': 18})
plt.ylabel('Probability')
plt.xlabel('PM Time')
plt.xticks([-60, -30, 0, 30, 60, 90, 120], sleep_labels)

print('The probability of sleep increases to above 50% at 10:{} PM.'.format(
    int(time_est[np.where(sleep_est > 0.5)[0][0]][0])))

# %%

colors = ["#348ABD", "#A60628", "#7A68A6"]
cmap = matplotlib.colors.LinearSegmentedColormap.from_list("BMH", colors)
figsize(12, 6)
probs = sleep_trace['p']

plt.scatter(time, probs.mean(axis=0), cmap=cmap,
            c=probs.mean(axis=0), s=50)
plt.title('Probability of Sleep as Function of Time')
plt.xlabel('PM Time')
plt.ylabel('Probability')
plt.xticks([-60, -30, 0, 30, 60, 90, 120], sleep_labels)

print('10:00 PM probability of being asleep: {:.2f}%.'.
      format(100 * logistic(0, beta_est, alpha_est)))
print('9:30  PM probability of being asleep: {:.2f}%.'.
      format(100 * logistic(-30, beta_est, alpha_est)))
print('10:30 PM probability of being asleep: {:.2f}%.'.
      format(100 * logistic(30, beta_est, alpha_est)))

sleep_all_est = logistic(time_est.T, beta_samples, alpha_samples)
quantiles = stats.mstats.mquantiles(sleep_all_est, [0.025, 0.975], axis=0)

# %%

plt.fill_between(time_est[:, 0], *quantiles, alpha=0.6,
                 color='slateblue', label='95% CI')
plt.plot(time_est, sleep_est, lw=2, ls='--',
         color='black', label="average posterior \nprobability of sleep")
plt.xticks([-60, -30, 0, 30, 60, 90, 120], sleep_labels)
plt.scatter(time, sleep_obs, edgecolor='skyblue', s=50, alpha=0.1)
plt.legend(prop={'size': 14})
plt.xlabel('PM Time')
plt.ylabel('Probability')
plt.title('Posterior Probabilty with 95% CI')


def sleep_posterior(time_offset, time):
    figsize(16, 8)
    prob = logistic(time_offset, beta_samples, alpha_samples)
    plt.hist(prob, bins=100, histtype='step', lw=4)
    plt.title('Probability Distribution for Sleep at %s PM' % time)
    plt.xlabel('Probability of Sleep')
    plt.ylabel('Samples')
    plt.show()


# %%

sleep_posterior(0, '10:00')

# %%

sleep_posterior(-30, '9:30')

# %%

sleep_posterior(30, '10:30')

# %%

print('Most likely alpha parameter: {:.6f}.'.format(alpha_est))
print('Most likely beta  parameter: {:.6f}.'.format(beta_est))

figsize(12, 6)

# Plot alpha trace
plt.subplot(211)
plt.title(r'Trace of $\alpha$')
plt.plot(alpha_samples, color='darkred')
plt.xlabel('Samples')
plt.ylabel('Parameter')

# Plot beta trace
plt.subplot(212)
plt.title(r'Trace of $\beta$')
plt.plot(beta_samples, color='b')
plt.xlabel('Samples')
plt.ylabel('Parameter')
plt.tight_layout(h_pad=0.8)

figsize(20, 12)
pm.traceplot(sleep_trace, ['alpha', 'beta'])

# %%

pm.autocorrplot(sleep_trace, ['alpha', 'beta'])

# Sort the values by time offset
wake_data.sort_values('time_offset', inplace=True)

# Time is the time offset
time = np.array(wake_data.loc[:, 'time_offset'])

# Observations are the indicator
wake_obs = np.array(wake_data.loc[:, 'indicator'])

with pm.Model() as wake_model:
    # Create the alpha and beta parameters
    alpha = pm.Normal('alpha', mu=0.0, tau=0.01, testval=0.0)
    beta = pm.Normal('beta', mu=0.0, tau=0.01, testval=0.0)

    # Create the probability from the logistic function
    p = pm.Deterministic('p', 1. / (1. + tt.exp(beta * time + alpha)))

    # Create the bernoulli parameter which uses the observed data
    observed = pm.Bernoulli('obs', p, observed=wake_obs)

    # Starting values are found through Maximum A Posterior estimation
    # start = pm.find_MAP()

    # Using Metropolis Hastings Sampling
    step = pm.Metropolis()

    # Sample from the posterior using the sampling method
    # wake_trace = pm.sample(N_SAMPLES, step=step, njobs=2)
    wake_trace = pm.sample(N_SAMPLES, step=step, cores=1)

# Extract the alpha and beta samples
alpha_samples = wake_trace["alpha"][5000:, None]
beta_samples = wake_trace["beta"][5000:, None]

# Time values for probability prediction
time_est = np.linspace(time.min() - 15, time.max() + 15, int(1e3))[:, None]

# Take most likely parameters to be mean values
alpha_est = alpha_samples.mean()
beta_est = beta_samples.mean()

# Probability at each time using mean values of alpha and beta
wake_est = logistic(time_est, beta=beta_est, alpha=alpha_est)

figsize(16, 6)

plt.plot(time_est, wake_est, color='darkred',
         lw=3, label="average posterior \nprobability of wake")
plt.scatter(time, wake_obs, edgecolor='r', facecolor='r',
            s=50, alpha=0.05, label='obs')
plt.title('Posterior Probability of Wake with %d Samples' % N_SAMPLES)
plt.legend(prop={'size': 14})
plt.ylabel('Probability')
plt.xlabel('AM Time')
plt.xticks([-60, -30, 0, 30, 60, 90, 120], wake_labels)

print('The probability of being awake passes 50% at 6:{} AM.'.format(
    int(time_est[np.where(wake_est < 0.5)][0])))

# %%

colors = ["#348ABD", "#A60628", "#7A68A6"]
cmap = matplotlib.colors.LinearSegmentedColormap.from_list("BMH", colors)
figsize(12, 6)
probs = wake_trace['p']

plt.scatter(time, probs.mean(axis=0), cmap=cmap,
            c=probs.mean(axis=0), s=50)
plt.title('Probability of Sleep as Function of Time')
plt.xlabel('AM Time')
plt.ylabel('Probability')
plt.xticks([-60, -30, 0, 30, 60, 90, 120], wake_labels)

print('Probability of being awake at 5:30 AM: {:.2f}%.'.
      format(100 - (100 * logistic(-30, beta=beta_est, alpha=alpha_est))))
print('Probability of being awake at 6:00 AM: {:.2f}%.'.
      format(100 - (100 * logistic(0, beta=beta_est, alpha=alpha_est))))
print('Probability of being awake at 6:30 AM: {:.2f}%.'.
      format(100 - (100 * logistic(30, beta=beta_est, alpha=alpha_est))))

raw_data = pd.read_csv('data/sleep_wake.csv')
raw_data['length'] = 8 - (raw_data['Sleep'] / 60) + (raw_data['Wake'] / 60)
duration = raw_data['length']

# %%

figsize(10, 8)
plt.hist(duration, bins=20, color='darkred')
plt.xlabel('Hours')
plt.title('Length of Sleep Distribution')
plt.ylabel('Observations')

a = 3
fig, ax = plt.subplots(1, 1)
x = np.linspace(6, 12, int(1e3))

figsize(10, 8)
plt.hist(duration, bins=20, color='darkred', density=True, stacked=True)
plt.xlabel('Hours')
plt.title('Length of Sleep Distribution with Skewed PDF')
plt.ylabel('Observations')
plt.plot(x, stats.skewnorm.pdf(x, a, loc=7.4, scale=1), 'r-',
         lw=3, label='skewnorm pdf')

# %%

with pm.Model() as duration_model:
    # Three parameters to sample
    alpha_skew = pm.Normal('alpha_skew', mu=0, tau=0.5, testval=3.0)
    mu_ = pm.Normal('mu', mu=0, tau=0.5, testval=7.4)
    tau_ = pm.Normal('tau', mu=0, tau=0.5, testval=1.0)

    # Duration is a deterministic variable
    duration_ = pm.SkewNormal('duration', alpha=alpha_skew, mu=mu_,
                              sd=1 / tau_, observed=duration)

    # Metropolis Hastings for sampling
    step = pm.Metropolis()
    # duration_trace = pm.sample(N_SAMPLES, step=step)
    duration_trace = pm.sample(N_SAMPLES, step=step, cores=1)

# %%

# Extract the most likely estimates from the sampling
alpha_skew_samples = duration_trace['alpha_skew'][5000:]
mu_samples = duration_trace['mu'][5000:]
tau_samples = duration_trace['tau'][5000:]

alpha_skew_est = alpha_skew_samples.mean()
mu_est = mu_samples.mean()
tau_est = tau_samples.mean()

x = np.linspace(6, 12, 1000)
y = stats.skewnorm.pdf(x, a=alpha_skew_est, loc=mu_est, scale=1 / tau_est)
plt.plot(x, y, color='forestgreen')
plt.fill_between(x, y, color='forestgreen', alpha=0.2)
plt.xlabel('Hours')
plt.ylabel('Probability')
plt.title('Posterior Distribution for Duration of Sleep')
plt.vlines(x=x[np.argmax(y)], ymin=0, ymax=y.max(),
           linestyles='--', linewidth=2, color='red',
           label='Most Likely Duration')

print('The most likely duration of sleep is {:.2f} hours.'.format(x[np.argmax(y)]))

print('Probability of at least 6.5 hours of sleep = {:.2f}%.'.
      format(100 * (1 - stats.skewnorm.cdf(6.5, a=alpha_skew_est, loc=mu_est, scale=1 / tau_est))))
print('Probability of at least 8.0 hours of sleep = {:.2f}%.'.
      format(100 * (1 - stats.skewnorm.cdf(8.0, a=alpha_skew_est, loc=mu_est, scale=1 / tau_est))))
print('Probability of at least 9.0 hours of sleep = {:.2f}%.'.
      format(100 * (1 - stats.skewnorm.cdf(9.0, a=alpha_skew_est, loc=mu_est, scale=1 / tau_est))))

x = np.linspace(6, 12, 1000)
y = stats.skewnorm.pdf(x, a=alpha_skew_est, loc=mu_est, scale=1 / tau_est)
figsize(10, 8)
# Plot the posterior distribution
plt.plot(x, y, color='forestgreen',
         label='Model', lw=3)
plt.fill_between(x, y, color='forestgreen', alpha=0.2)

# Plot the observed values
plt.hist(duration, bins=10, color='red', alpha=0.8,
         label='Observed', normed=True)
plt.xlabel('Hours')
plt.ylabel('Probability')
plt.title('Duration Model')
plt.vlines(x=x[np.argmax(y)], ymin=0, ymax=y.max(),
           linestyles='--', linewidth=2, color='k',
           label='Most Likely Duration')
plt.legend(prop={'size': 12})
