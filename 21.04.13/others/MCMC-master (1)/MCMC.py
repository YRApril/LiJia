# encoding: utf-8

import numpy as np
import scipy
import scipy.stats
import matplotlib as mpl
import matplotlib.pyplot as plt

"""

# From Scratch: Bayesian Inference, Markov Chain Monte Carlo and Metropolis Hastings, in python
来自Scratch：贝叶斯推理，Markov Chain Monte Carlo和Metropolis Hastings，使用python



In one of the courses during my data science degree, I came across a technique called Markov Chain Monte Carlo, or as it's more commonly referred to, MCMC. The description for this method stated something along the lines of: MCMC is a class of techniques for sampling from a probability distribution and can be used to estimate the distribution of parameters given a set of observations.
在我的数据科学学位课程中的一门课程中，我遇到了一种名为“马尔可夫链蒙特卡洛”（Markov Chain Monte Carlo）的技术，或者通常被称为MCMC的技术。 此方法的描述遵循以下思路：MCMC是一类用于从概率分布中采样的技术，在给定一组观察值的情况下，可用于估计参数的分布。

Back then, I did not think much of it. I thought, "oh it's just another sampling technique", and I decided I'd read on it when I'd practically need it. This need never emerged, or perhaps it did and I wrongly used something else.
那时，我并没有考虑太多。 我以为“哦，这只是另一种采样技术”，所以我决定在实际需要时就读一读。 这种需求从未出现过，或者确实发生了，而我错误地使用了其他东西。

 

## So why the interest now?
那么为什么现在有兴趣呢？

Recently, I have seen a few discussions about MCMC and some of its implementations, specifically the Metropolis-Hastings algorithm and the PyMC3 library. $Markov \: Chain\: Monte\: Carlo\: in\: Python\: - \:A \:Complete\: Real$-$World \:Implementation$ was the article that caught my attention the most. In it, William Koehrsen explains how he was able to learn the approach by applying it to a real world problem: to estimate the parameters of a logistic function that represents his sleeping patterns.
最近，我看到了一些有关MCMC及其实现的讨论，特别是Metropolis-Hastings算法和PyMC3库。Markov Chain Monte Carlo in  Python  -- A Complete  Real - World  Implementation 是吸引我最多注意力的文章。 在其中，威廉·科赫森（William 
 Koehrsen）解释了如何通过将其应用于实际问题来学习该方法：估算代表他的睡眠方式的逻辑函数的参数。

 

Mr. Koehrsen uses the PyMC3 implementation of the Metropolis-Hastings algorithm to estimate $\beta$ and $\alpha$, thus computing the entire parameter space, and deriving the most likely logistic model.
Koehrsen先生使用Metropolis-Hastings算法的PyMC3实现来估算$ \ beta $和$ \ alpha $，从而计算整个参数空间，并得出最可能的逻辑模型。

 

## So why am I talking about all that?
那我为什么要谈论所有这些呢？

I this article, I propose to implement from scratch, my own version of the Metropolis-Hastings algorithm to find parameter distributions for a dummy data example and then of a real world problem.
在本文中，我建议从头开始实现我自己的Metropolis-Hastings算法版本，以查找虚拟数据示例的参数分布，然后找到实际问题。

I figured that if I get my hands dirty, I might finally be able to understand it. I will only use numpy to implement the algorithm, and matplotlib to draw pretty things. Alternatively, scipy can be used to compute the density functions, but I will also show how to implement them using numpy.
我认为，如果我弄脏了手，也许我终于可以理解它了。 我将仅使用numpy来实现该算法，并使用matplotlib来绘制漂亮的东西。 另外，scipy可用于计算密度函数，但我还将展示如何使用numpy实现它们。

 

## Flow of the article:

    * At first, I will introduce Bayesian inference, MCMC-MH and their mathematical components.
    * Second, I will explain the algorithm using dummy data.
    * Third, I will apply it to a real world problem.
    
    文章流程：

     *首先，我将介绍贝叶斯推理，MCMC-MH及其数学组成部分。
     *第二，我将解释使用伪数据的算法。
     *第三，我将其应用于现实世界中的问题。

 

# Part 1: Bayesian inference, Markov Chain Monte Carlo, and Metropolis-Hastings
第1部分：贝叶斯推理，马尔可夫链蒙特卡洛和大都会哈斯丁

 

## A bird's eye view on the philosophy of probabilities
鸟瞰概率哲学

In order to talk about Bayesian inference and what follows it, I shall first explain what the Bayesian view of probability is, and situate it within its historical context
为了讨论贝叶斯推理及其后续内容，我将首先解释贝叶斯概率论的观点，并将其置于其历史背景下。

### Frequentist vs Bayesian thinking
频繁主义者与贝叶斯思想

There are two major interpretations to probabilities: Bayesian and Frequentist.
对概率有两种主要的解释：贝叶斯和频率论。 

From a **Frequentist's** perspective, probabilities represent long term frequencies with which events occur. A frequentist can say that the probability of having tails from a coin toss is equal to 0.5 *on the long run*. Each new experiment, can be considered as one of an infinite sequence of possible repetitions of the same experiment. The idea is that there is *no* belief in a frequentist's view of probability. The probability of event $x$ happening out of n trials is equal to the following frequency: $P(x)=\dfrac{n_x}{n}$, and the true probability is reached when $n->\infty$. Frequentists will never say "I am 45% (0.45) sure that there is lasagna for lunch today", since this does not happen on the long run. Commonly, a frequentist approach is referred to as the *objective* approach since there is no expression of belief and/or prior events in it.
从“频率论者”的角度来看，概率代表事件发生的长期频率。 一位常客可以说，从长远来看，抛硬币产生尾巴的概率等于0.5。 每个新实验都可以视为同一实验可能重复的无限序列之一。 这个想法是*没有*相信常客的概率观点。 n次试验中事件$ x $发生的概率等于以下频率：$ P（x）= \ dfrac {n_x} {n} $，并且当$ n-> \ infty $时才达到真实概率。 频繁的人永远不会说“我确定今天有45％（0.45）的午餐可以吃千层面”，因为从长远来看这不会发生。 通常，常客方法被称为“客观方法”，因为其中没有表达信念和/或先前事件的表达。

On the other hand, in **Bayesian** thinking, probabilities are treated as an expression of belief. Therefore it is perfectly reasonable for a Bayesian to say "I am 50% (0.5) sure that there is lasagna for lunch today". By combining *prior* beliefs and current events (the *evidence*) one can compute the *posterior*, the belief that there is lasagna today. The idea behind Bayesian thinking is to keep updating the beliefs as more evidence is provided. Since this approach deals with belief, it is usually referred to as the *subjective* view on probability.
另一方面，在贝叶斯（Bayesian）思维中，概率被视为信念的一种表达。 因此，贝叶斯说“我有50％（0.5）确信今天午餐有烤宽面条”是完全合理的。 通过将*先前*的信念和当前事件（*证据*）相结合，可以计算出*后**，即今天存在千层面。 贝叶斯思想背后的思想是随着提供更多证据而不断更新信念。 由于这种方法处理信念，因此通常被称为概率的“主观”观点。

### Bayesian inference
贝叶斯推理





In the philosophy of decision making, Bayesian inference is closely related to Bayesian probability, in the sense that it manipulates priors, evidence, and likelihood to compute the posterior. Given some event B, what is the probability that event A occurs?. This is answered by Bayes' famous formula: $P(A/B)=\dfrac{P(B/A)P(A)}{P(B)}$
在决策的哲学中，贝叶斯推理与贝叶斯概率密切相关，因为贝叶斯推理会操纵先验，证据和可能性来计算后验。 给定某些事件B，事件A发生的概率是多少？ 贝叶斯的著名公式回答了这一问题：

* $P(A/B)$ is the **posterior**. What we wish to compute.
$ P（A / B）$是后验**。 我们希望计算的。

* $P(B/A)$ is the **likelihood**. Assuming A occured, how likely is B.
$ P（B / A）$是“可能性”。 假设发生了A，那么B发生的可能性是多少。

* $P(A)$ is the **prior**. How likely the event $A$ is regardless of evidence.
$ P（A）$是优先级**。 无论证据如何，事件$ A $的可能性是多少。

* $P(B)$ is the **evidence**. How likely the evidence $B$ is regardless of the event.
$ P（B）$是证据**。 不论事件如何，证据$ B $的可能性有多大。

In our case, we are mostly interested in the specific formulation of Bayes' formula:
在我们的案例中，我们对贝叶斯公式的特定公式最感兴趣：



We would like to find the most likely distribution of $\theta$, the parameters of the model explaining the data, D.
我们想找到$ \ theta $最可能的分布，即解释数据D的模型参数。


Computing some of these probabilities can be tedious, especially the evidence $P(D)$. Also, other problems can arise such as those of ensuring conjugacy, which I will not dive into in this article. Luckily, some techniques, namely MCMC, allow us to sample from the posterior, and a draw distributions over our parameters without having to worry about computing the evidence, nor about conjugacy.
计算其中一些概率可能是乏味的，尤其是证据$ P（D）$。 此外，还会出现其他问题，例如确保共轭的问题，我将不在本文中介绍。 幸运的是，某些技术（即MCMC）使我们可以从后验中取样，并在我们的参数上绘制分布，而不必担心计算证据或共轭。

### Markov Chain Monte Carlo
马尔可夫链蒙特卡洛


MCMC allows us to draw samples from any distribution that we can't sample from directly. It can be used to sample from the posterior distribution over parameters.
MCMC允许我们从无法直接采样的任何分布中抽取采样。 它可用于从参数的后验分布中采样。

It has seen much success in many applications, such as computing the distribution of parameters, given a set of observations and some prior belief, and also computing high dimensional integrals in physics and in digital communications.
它在许多应用中都取得了很大的成功，例如计算参数的分布，给出一组观察结果和一些先验的信念，以及在物理和数字通信中计算高维积分。

Bottom line: **It can be used to compute the distribution over the parameters, given a set of observations and a prior belief.**
底线：**在给定一组观察值和先验信念的情况下，可用于计算参数的分布。**

### Metropolis-Hastings
大都会-哈丁斯

MCMC is a class of methods. Metropolis-Hastings is a specific implementation of MCMC. It works well in high dimensional spaces as opposed to Gibbs sampling and rejection sampling.
MCMC是一类方法。 Metropolis-Hastings是MCMC的特定实现。 与Gibbs采样和拒绝采样相反，它在高维空间中效果很好。

This technique requires a simple distribution called the **proposal distribution** (Which I like to call **transition model**) $Q(\theta^\prime/\theta)$ to help draw samples from an intractable posterior distribution $P(\Theta=\theta/D)$. 


Metropolis-Hastings uses $Q$ to randomly walk in the distribution space, accepting or rejecting jumps to new positions based on how likely the sample is. This "memoriless" random walk is the "Markov Chain" part of MCMC.

The "likelihood" of each new sample is decided by a function $f$ . That's why $f$ must be proportional to the posterior we want to sample from. f is commonly chosen to be a probability density function that expresses this proportionality.

To get a new position of the parameter, just take our current one $\theta$, and propose a new one $\theta^\prime$, that is a random sample drawn from $Q(\theta^\prime/\theta)$. Often this is a symmetric distribution. For instance, a normal distribution with mean $\theta$ and some standard deviation 


To decide if $\theta^\prime$ is to be accepted or rejected, the following ratio must be computed for each new value of $\theta^\prime$: 

This means that if a θ' is more likely than the current θ, then we always accept θ'. If it is less likely than the current θ, then we might accept it or reject it randomly with decreasing probability, the less likely it is.
这意味着如果θ'比当前θ的可能性更大，那么我们总是会接受θ'。 如果它比当前θ的可能性小，则我们可以接受它，也可以以降低的概率随机拒绝它，它的可能性就较小。

*Note: The prior components are often crossed if there is no preference or restrictions on the parameters.*


#### Metropolis-Hastings Algorithm:

"""

"""
Part 2: Dummy data example
虚拟数据示例
## Step 1: Data generation
数据生成

We generate 30,000 samples from a normal distribution with $\mu$ = 10, and $\sigma$= 3, but we can only observe 1000 of them.
我们从正态分布中生成30,000个样本，其中$ \ mu $ = 10，而$ \ sigma $ = 3，但我们只能观察其中的1000个

"""

mod1 = lambda t: np.random.normal(10, 3, t)  # 匿名函数，生成指定数量的正态分布

# Form a population of 30,000 individual, with average=10 and scale=3
population = mod1(30000)  # 形成30,000个人的人口，平均数= 10，规模= 3
# Assume we are only able to observe 1,000 of these individuals.
observation = population[np.random.randint(0, 30000, 1000)]  # 随机取其中1000个

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(1, 1, 1)
ax.hist(observation, bins=35, )
ax.set_xlabel("Value")
ax.set_ylabel("Frequency")
ax.set_title(
    "Figure 1: Distribution of 1000 observations sampled from a population of 30,000 with $\mu$=10, $\sigma$=3")
mu_obs = observation.mean()
print(mu_obs)

"""
## Step 2: What do we want?

We would like to find a distribution for $\sigma_{obs}$ using the 1000 observed samples. Those of you who are adept at mathematics will say that there is a formula for computing $\sigma$ ( $\sigma=\sqrt{\dfrac{1}{n}\sum_i^n(d_i-\mu)^2}$)! Why do we want to sample and whatnot?? Well, this is just a dummy data example, the real problem is in part 3, where the parameters cannot be computed directly. Plus here, we are not trying to find *a* value for $\sigma$, but rather, we are trying to compute a distribution of the possible values of $\sigma$.

## Step 3: Define the PDF and the transition model.

From Figure 1, we can see that the data is normally distributed. The mean can be easily computed by taking the average of the values of the 1000 samples. By doing that, we get for example $\mu_{obs}=9.8$.

### For the transition model/ proposal distribution:
I have no specific distribution in mind, so I will choose a simple one: the Normal distribution!

\begin{equation} Q(\sigma_{new} / \sigma_{current}) = N(\mu=\sigma_{current},\sigma'=1) \end{equation}

Note that $\sigma'$ is unrelated to $\sigma_{new}$ and $\sigma_{current}$. It simply specifies the standard deviation of the parameter space. It can be any value desired. It only affects the convergence time of the algorithm.

### For the PDF: 
Since f should be proportional to the posterior, we choose f to be the following Probability Density Function (PDF), for each data point $d_i$ in the data D:

\begin{equation} f(d_i/ \mu,\sigma^2) = \dfrac{1}{\sqrt{2\pi\sigma^2}}e^{-\dfrac{(d_i-\mu)^2}{2\sigma^2}} \end{equation}

In our case, $\theta$ is made up of two values: $[\mu,\sigma]$, and that $\mu$ is a constant, $\mu = \mu_{obs}$.

## Step 4: Define when we accept or reject $\sigma_{new}$: 
We accept $\sigma_{new}$ if:

$\dfrac{Likelihood(D/\mu_{obs},\sigma_{new})*prior(\mu_{obs},\sigma_{new})}{Likelihood(D/\mu_{obs},\sigma_{current})*prior(\mu_{obs},\sigma_{current})}>1     \quad \quad \quad \quad \quad      (1)$

If this ratio is smaller or equal to 1, then we compare it to a uniformly generated random number in the closed set [0,1]. If the ratio is larger than the random number, we accept $\sigma_{new}$, otherwise we reject it.

*Note: Since we will be computing this ratio to decide which parameters should be accepted, it is imperative to make sure that the adopted function $f$  is proportional to the posterior itself, $P(\sigma/ D,\mu)$, which in that case is verified. ($f$ is the PDF of P)*


## Step 5: Define the prior and the likelihood:
### For the Prior $P(\theta)$ which we can alternatively note $P(\sigma)$ since $\mu$ is constant:
We don't have any preferences for the values that $\sigma_{new}$ and $\sigma_{current}$ can take. The only thing worth noting is that they should be positive. Why? Intuitively, the standard deviation measures dispersion. Dispersion is a distance, and distances cannot be negative. Mathematically, $\sigma=\sqrt{\dfrac{1}{n}\sum_i^n(d_i-\mu)^2}$, and the square root of a number cannot be negative. We strictly enforce this in the prior.


### For the likelihood :
The total likelihood for a set of observation $D$ is: $Likelihood(D/\mu_{obs},\sigma_{a}) = \prod_i^n f(d_i/\mu_{obs},\sigma_{a}) $, where $a=new \: or \: current$.

In our case, we will log both the prior and the likelihood function. Why log? Simply because it helps with numerical stability, i.e. multiplying thousands of small values (probabilities, likelihoods, etc..) can cause an underflow in the system's memory, and the log is a perfect solution because it transforms multiplications to additions and small positive numbers into non-small negative numbers.

Therefore our acceptance condition from equation $(1)$ becomes:

Accept $\sigma_{new}$ if:





$\quad \quad \quad \quad \quad Log(Likelihood(D/\mu_{obs},\sigma_{new})) + Log(prior(\mu_{obs},\sigma_{new})) - (Log(Likelihood(D/\mu_{obs},\sigma_{current}))+$

$Log(prior(\mu_{obs},\sigma_{current})))>0$

 $\quad$

 Equivalent to:
 
 $\sum_i^nLog(f(d_i/\mu_{obs},\sigma_{new})) + Log(prior(\mu_{obs},\sigma_{new})) - \sum_i^nLog(f(d_i/\mu_{obs},\sigma_{current}))-Log(prior(\mu_{obs},\sigma_{current}))>0$
 
 $\quad$
 
 
  Equivalent to:
  
  $\sum_i^nLog(f(d_i/\mu_{obs},\sigma_{new})) + Log(prior(\mu_{obs},\sigma_{new})) > \sum_i^nLog(f(d_i/\mu_{obs},\sigma_{current}))+Log(prior(\mu_{obs},\sigma_{current}))$
  
   $\quad$
  
  Equivalent to: 
  
  $\sum_i^n -nLog(\sigma_{new}\sqrt{2\pi})-\dfrac{(d_i-\mu_{obs})^2}{2\sigma_{new}^2} + Log(prior(\mu_{obs},\sigma_{new})) \quad > $
  
 $ \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad(2)$
 $ \quad \quad \quad \quad \quad \sum_i^n -nLog(\sigma_{current}\sqrt{2\pi})-\dfrac{(d_i-\mu_{obs})^2}{2\sigma_{current}^2}+Log(prior(\mu_{obs},\sigma_{current}))  $
  
  This form can be reduced even more by taking the square root and the multiplication out of the log.


"""

# The tranistion model defines how to move from sigma_current to sigma_new
transition_model = lambda x: [x[0],
                              np.random.normal(x[1], 0.5, (1,))[0]]  # 定义了如何从sigma_current迁移到sigma_new的过渡模型


def prior(x):
    # x[0] = mu, x[1]=sigma (new or current)
    # returns 1 for all valid values of sigma. Log(1) =0, so it does not affect the summation.
    # 对于sigma的所有有效值返回1。 Log（1）= 0，因此它不影响求和。
    # returns 0 for all invalid values of sigma (<=0). Log(0)=-infinity, and Log(negative number) is undefined.
    # 对于所有无效的sigma（<= 0）值返回0。 Log（0）=-无穷大，并且Log（负数）未定义。
    # It makes the new sigma infinitely unlikely.
    # 这使新的sigma无限可能出现。
    if (x[1] <= 0):
        return 0
    return 1


# Computes the likelihood of the data given a sigma (new or current) according to equation (2)
# 根据公式（2）计算给定sigma（新的或当前的）的数据的可能性
def manual_log_like_normal(x, data):
    # x[0]=mu, x[1]=sigma (new or current)
    # data = the observation
    return np.sum(-np.log(x[1] * np.sqrt(2 * np.pi)) - ((data - x[0]) ** 2) / (2 * x[1] ** 2))


# Same as manual_log_like_normal(x,data), but using scipy implementation. It's pretty slow.
# 与manual_log_like_normal（x，data）相同，但是使用scipy实现。 非常慢。
def log_lik_normal(x, data):
    # x[0]=mu, x[1]=sigma (new or current)
    # data = the observation
    return np.sum(np.log(scipy.stats.norm(x[0], x[1]).pdf(data)))


# Defines whether to accept or reject the new sample
# 定义是接受还是拒绝新样本
def acceptance(x, x_new):
    if x_new > x:
        return True
    else:
        accept = np.random.uniform(0, 1)
        # Since we did a log likelihood, we need to exponentiate in order to compare to the random number
        # less likely x_new are less likely to be accepted
        # 因为我们做了对数似然，所以我们需要取幂以与随机数进行比较
        # x_new被接受的可能性较小
        return (accept < (np.exp(x_new - x)))


def metropolis_hastings(likelihood_computer, prior, transition_model, param_init, iterations, data,
                        acceptance_rule):
    # likelihood_computer(x,data): returns the likelihood that these parameters generated the data
    # transition_model(x): a function that draws a sample from a symmetric distribution and returns it
    # param_init: a starting sample
    # iterations: number of accepted to generated
    # data: the data that we wish to model
    # acceptance_rule(x,x_new): decides whether to accept or reject the new sample
    # likelihood_computer(x,data)：返回这些参数生成数据的可能性
    # transition_model（x）：从对称分布中抽取样本并返回样本的函数
    # param_init：一个初始示例
    # iterations：接受到生成的次数
    # data：我们希望建模的数据
    # accepting_rule（x，x_new）：决定是接受还是拒绝新样本
    x = param_init  # 初试样例
    accepted = []  # 接收
    rejected = []  # 拒绝
    for i in range(iterations):
        x_new = transition_model(x)
        x_lik = likelihood_computer(x, data)
        x_new_lik = likelihood_computer(x_new, data)
        if (acceptance_rule(x_lik + np.log(prior(x)), x_new_lik + np.log(prior(x_new)))):
            x = x_new
            accepted.append(x_new)
        else:
            rejected.append(x_new)

    return np.array(accepted), np.array(rejected)


"""
## Step 6: Run the algorithm with initial parameters and collect accepted and rejected samples
使用初始参数运行算法，并收集接受和拒绝的样本
"""

accepted, rejected = metropolis_hastings(manual_log_like_normal, prior, transition_model, [mu_obs, 0.1],
                                         50000, observation, acceptance)

"""
### The algorithm accepted 8803 samples (which might be different on each new run). The last 10 samples contain the following  values for $\sigma$:
该算法接受了8803个样本（每次新运行可能会有所不同）。 最后的10个样本包含$ \ sigma $的以下值：
"""
print(accepted[-10:, 1])

print(accepted.shape)

"""

So, starting from an initial σ of 0.1, the algorithm converged pretty quickly to the expected value of 3. That said, it's only sampling in a 1D space…. so it's not very surprising.

因此，从初始σ0.1开始，该算法很快收敛到期望值3。也就是说，它只是在一维空间中进行采样……。 所以这并不奇怪。


### We consider the initial 25% of the values of $\sigma$ to be "burn-in", so we drop them.
我们认为$ \ sigma $的最初25％的值是“烙印”，因此我们将其删除。

### Let's visualize the trace of  $\sigma$ and the histogram of the trace.
让我们可视化$ \ sigma $的轨迹和轨迹的直方图。

"""

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(2, 1, 1)

ax.plot(rejected[0:50, 1], 'rx', label='Rejected', alpha=0.5)
ax.plot(accepted[0:50, 1], 'b.', label='Accepted', alpha=0.5)
ax.set_xlabel("Iteration")
ax.set_ylabel("$\sigma$")
ax.set_title("Figure 2: MCMC sampling for $\sigma$ with Metropolis-Hastings. First 50 samples are shown.")
ax.grid()
ax.legend()

ax2 = fig.add_subplot(2, 1, 2)
to_show = -accepted.shape[0]
ax2.plot(rejected[to_show:, 1], 'rx', label='Rejected', alpha=0.5)
ax2.plot(accepted[to_show:, 1], 'b.', label='Accepted', alpha=0.5)
ax2.set_xlabel("Iteration")
ax2.set_ylabel("$\sigma$")
ax2.set_title("Figure 3: MCMC sampling for $\sigma$ with Metropolis-Hastings. All samples are shown.")
ax2.grid()
ax2.legend()

fig.tight_layout()
print(accepted.shape)

show = int(-0.75 * accepted.shape[0])
hist_show = int(-0.75 * accepted.shape[0])

fig = plt.figure(figsize=(20, 10))
ax = fig.add_subplot(1, 2, 1)
ax.plot(accepted[show:, 1])
ax.set_title("Figure 4: Trace for $\sigma$")
ax.set_ylabel("$\sigma$")
ax.set_xlabel("Iteration")
ax = fig.add_subplot(1, 2, 2)
ax.hist(accepted[hist_show:, 1], bins=20, density=True)
ax.set_ylabel("Frequency (normed)")
ax.set_xlabel("$\sigma$")
ax.set_title("Figure 5: Histogram of $\sigma$")
fig.tight_layout()

ax.grid("off")


"""

### The most likely value for $\sigma$ is around 3.1. This is a bit more than the original value of 3.0. The difference is due to us observing only 3.33% of the original population (1,000 out of 30,000) 
$ \ sigma $的最可能值约为3.1。 这比原始值3.0多一点。 差异是由于我们只观察了原始人口的3.33％（30,000中的1,000）

## Predictions:
预测
First, we average the last 75% of accepted samples of σ, and we generate 30,000 random individuals from a normal distribution with μ=9.8 and σ=3.05 (the average of the last 75% of accepted samples) which is actually better than the most likely value of 3.1.
首先，我们对σ的最后75％的样本取平均值，然后从正态分布中生成30,000个随机个体，其中μ= 9.8和σ= 3.05（最后75％的被接受样本的平均值）实际上要好于 最可能的值为3.1。

"""

mu = accepted[show:, 0].mean()
sigma = accepted[show:, 1].mean()
print(mu, sigma)
model = lambda t, mu, sigma: np.random.normal(mu, sigma, t)
observation_gen = model(population.shape[0], mu, sigma)
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(1, 1, 1)
ax.hist(observation_gen, bins=70, label="Predicted distribution of 30,000 individuals")
ax.hist(population, bins=70, alpha=0.5, label="Original values of the 30,000 individuals")
ax.set_xlabel("Mean")
ax.set_ylabel("Frequency")
ax.set_title("Figure 6: Posterior distribution of predicitons")
ax.legend()
