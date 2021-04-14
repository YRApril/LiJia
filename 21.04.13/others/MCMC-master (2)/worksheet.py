import numpy as np
import matplotlib.pyplot as plt

name = "YOUR NAME HERE"
print("Hello {0}!".format(name))

"""
If this works, the output should greet you without throwing any errors. If so, that's pretty much all we need so let's get started with some MCMC!
如果这样可行，输出应向您打招呼，而不会引发任何错误。 
如果是这样，那几乎就是我们所需要的，因此让我们开始使用一些MCMC！

## Dataset 1: Fitting a line to data
数据集1：将一条线拟合到数据

Today, we're going to implement the simplest possible MCMC algorithm but before we do that, we'll need some data to test our method with.
今天，我们将实现最简单的MCMC算法，但在此之前，我们将需要一些数据来测试我们的方法。

## Load the data
加载数据

I've generated a simulated dataset generated from a linear model with no uncertainties in the $x$ dimension and known Gaussian uncertainties in the $y$ dimension. These data are saved in the CSV file `linear.csv` included with this notebook.
我已经生成了从线性模型生成的模拟数据集，该模型在$ x $维度中没有不确定性，在$ y $维度中没有已知的高斯不确定性。 这些数据保存在此笔记本随附的CSV文件“ linear.csv”中。

First we'll need `numpy` and `matplotlib` so let's import them:
"""

# %matplotlib inline
from matplotlib import rcParams

rcParams["savefig.dpi"] = 100  # This makes all the plots a little bigger.

"""Now we'll load the datapoints and plot them. When you execute the following cell, you should see a plot of the data. If not, make sure that you run the import cell from above first.
现在，我们将加载数据点并绘制它们。 
当执行以下代码时，您应该看到数据图。 如果不是，请确保首先从上方运行导入单元。
"""

# Load the data from the CSV file.
# 从CSV文件加载数据。x轴坐标，y轴坐标，数据在y轴的误差范围
x, y, yerr = np.loadtxt("linear.csv", delimiter=",", unpack=True)

# Plot the data with error bars.
# 用误差线绘制数据
plt.errorbar(x, y, yerr=yerr, fmt=".k", capsize=0)
plt.xlim(0, 5)  # 设置x轴范围
# plt.show()
# exit()


"""
As I mentioned previously, it is pretty silly to use MCMC to solve this problem because the maximum likelihood and full posterior probability distribution (under infinitely broad priors) for the slope and intercept of the line are known analytically. Therefore, let's compute what the right answer should be before we even start. The analytic result for the posterior probability distribution is a 2-d Gaussian with mean
and covariance matrix
where
There are various functions in Python for computing this but I prefer to do it myself (it only takes a few lines of code!) and here it is:

正如我之前提到的，使用MCMC解决此问题是很愚蠢的，因为通过分析已知该线的斜率和截距的最大似然性和完整的后验概率分布（在无限大的先验条件下）。 因此，让我们在开始之前就计算出正确的答案。 后验概率分布的分析结果是二维的高斯平均值
和协方差矩阵
Python中有多种函数可用于计算此函数，但我更愿意自己做（它只需要几行代码！）：
"""

A = np.vander(x, 2)  # Take a look at the documentation to see what this function does!根据x生成两列范德蒙矩阵
ATA = np.dot(A.T, A / yerr[:, None] ** 2)  # 矩阵乘法，范德蒙矩阵的转置 * (范德蒙矩阵/误差矩阵)^2

w = np.linalg.solve(ATA, np.dot(A.T, y / yerr ** 2))  # 以矩阵形式解一个线性矩阵方程
V = np.linalg.inv(ATA)  # 矩阵求逆

"""
We'll save these results for later to compare them to the result computed using MCMC but for now, it's nice to take a look and see what this prediction looks like. To do this, we'll sample 24 slopes and intercepts from this 2d Gaussian and overplot them on the data.

This plot is a visualization of our posterior expectations for the *true* underlying line that generated these data. We'll reuse this plot a few times later to test the results of our code.

我们将保存这些结果供以后将其与使用MCMC计算的结果进行比较，但现在，很高兴了解一下此预测的外观。 为此，我们将采样24个坡度并从该2d高斯截取并在数据上叠加绘制它们

此图是对生成这些数据的“真实”底层线的后验预期的可视化。 稍后，我们将重用此图几次以测试代码的结果。
"""
plt.errorbar(x, y, yerr=yerr, fmt=".k", capsize=0)
for m, b in np.random.multivariate_normal(w, V, size=50):  # 生成一个多元正态分布矩阵,
    plt.plot(x, m * x + b, "g", alpha=0.1)  # 绘制拟合图像？
plt.xlim(0, 5)

# plt.show()
# exit()


"""

### The probabilistic model
概率模型

In order use MCMC to perform posterior inference on a model and dataset, we need a function that computes the value of the posterior probability given a proposed setting of the parameters of the model. For reasons that will become clear below, we actually only need to return a value that is *proportional* to the probability.
为了使用MCMC对模型和数据集进行后验推断，我们需要一个函数，该函数在给定建议的模型参数设置的情况下计算后验概率的值。 出于下面将变得清楚的原因，我们实际上只需要返回与概率成“比例”的值。

As discussed in a previous tutorial, the posterior probability for parameters $\mathbf{w} = (m,\,b)$ conditioned on a dataset $\mathbf{y}$ is given by

$$p(\mathbf{w} \,|\, \mathbf{y}) = \frac{p(\mathbf{y} \,|\, \mathbf{w}) \, p(\mathbf{w})}{p(\mathbf{y})}$$

where $p(\mathbf{y} \,|\, \mathbf{w})$ is the *likelihood* and $p(\mathbf{w})$ is the *prior*. For this example, we're modeling the likelihood by assuming that the datapoints are independent with known Gaussian uncertainties $\sigma_n$. This specifies a likelihood function:

$$p(\mathbf{y} \,|\, \mathbf{w}) = \prod_{n=1}^N \frac{1}{\sqrt{2\,\pi\,\sigma_n^2}} \,
\exp \left(-\frac{[y_n - f_\mathbf{w}(x_n)]^2}{2\,\sigma_n^2}\right)$$

where $f_\mathbf{w}(x) = m\,x + b$ is the linear model.

For numerical reasons, we will acutally want to compute the logarithm of the likelihood. In this case, this becomes:
由于数值原因，我们实际上将要计算似然的对数。 在这种情况下，它将变为：

$$\ln p(\mathbf{y} \,|\, \mathbf{w}) = -\frac{1}{2}\sum_{n=1}^N \frac{[y_n - f_\mathbf{w}(x_n)]^2}{\sigma_n^2} + \mathrm{constant} \quad.$$

In the following cell, replace the contents of the `lnlike_linear` function to implement this model. The function takes two values (`m` and `b`) as input and it should return the log likelihood (a single number) up to a constant. In this function, you can just use the globaly defined dataset `x`, `y` and `yerr`. For performance, I recommend using vectorized numpy operations (the key function will be `np.sum`).
在下面的单元格中，替换“ Inlike_linear”函数的内容以实现此模型。 该函数接受两个值（“ m”和“ b”）作为输入，并且应返回对数似然（单个数字），直到一个常数。 在此功能中，您可以仅使用全局定义的数据集“ x”，“ y”和“ yerr”。 为了提高性能，我建议使用向量化numpy操作（关键功能为`np.sum`）。
"""

#todo 替换“ Inlike_linear”函数的内容以实现此模型。 该函数接受两个值（“ m”和“ b”）作为输入，并且应返回对数似然（单个数字），直到一个常数。 在此功能中，您可以仅使用全局定义的数据集“ x”，“ y”和“ yerr”。 为了提高性能，我建议使用向量化numpy操作（关键功能为`np.sum`）。
def lnlike_linear(m, b):
    raise NotImplementedError("Delete this placeholder and implement this function")


p_1, p_2 = (0.0, 0.0), (0.01, 0.01)
ll_1, ll_2 = lnlike_linear(p_1[0],p_1[1]), lnlike_linear(p_2[0],p_2[1])
if not np.allclose(ll_2 - ll_1, 535.8707738280209):
    raise ValueError("It looks like your implementation is wrong!")
print("☺︎")



def lnprior_linear(m, b):
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


def lnlike_poisson(alpha, beta):
    raise NotImplementedError("Delete this placeholder and implement this function")


p_1, p_2 = (1000.0, -1.), (1500., -2.)
ll_1, ll_2 = lnlike_poisson(p_1), lnlike_poisson(p_2)
if not np.allclose(ll_2 - ll_1, 337.039175916):
    raise ValueError("It looks like your implementation is wrong!")
print("☺︎")


def lnprior_poisson(alpha, beta):
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
acc = float(np.any(np.diff(chain, axis=0), axis=1).sum()) / (len(chain) - 1)
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
for alpha, beta in chain[nburn + np.random.randint(len(chain) - nburn, size=50)]:
    plt.plot(xx, alpha * xx ** beta, "g", alpha=0.1)

# Format the figure.
plt.ylabel("number")
plt.xlabel("x");
