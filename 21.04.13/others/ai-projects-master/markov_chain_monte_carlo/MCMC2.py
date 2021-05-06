import pandas as pd
import numpy as np
from numpy.random.mtrand import logistic
from scipy import stats
import pymc3 as pm
import theano.tensor as tt
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from IPython.core.pylabtools import figsize
import matplotlib

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
# 解决图中负数不能正常显示的问题
matplotlib.rcParams['axes.unicode_minus'] = False

# --------------------------------------------导入数据--------------------------------------------------------
N_SAMPLES = 1000
sleep_data = pd.read_csv('data/sleep_data.csv')
wake_data = pd.read_csv('data/wake_data.csv')
sleep_labels = ['21:00', '21:30', '22:00', '22:30', '23:00', '23:30', '00:00']
wake_labels = ['5:00', '5:30', '6:00', '6:30', '7:00', '7:30', '8:00']


# --------------------------------------------睡眠数据分布_散点图-------------------------------------------------------
def test1():
    figsize(10, 6)
    _ = plt.scatter(sleep_data['time_offset'], sleep_data['indicator'],
                    s=60, alpha=0.01, facecolor='b', edgecolors='b')
    plt.yticks([0, 1], ['清醒', '入睡'], fontsize=15)
    plt.xlabel('晚上时间', fontsize=15)
    plt.title('睡眠数据分布', size=18)

    plt.xticks([-60, -30, 0, 30, 60, 90, 120], sleep_labels)
    plt.show()


# --------------------------------------------清醒数据分布_散点图--------------------------------------------------------
def test2():
    figsize(10, 6)
    plt.scatter(wake_data['time_offset'], wake_data['indicator'],
                s=50, alpha=0.01, facecolor='r', edgecolors='r')
    plt.yticks([0, 1], ['清醒', '入睡'], fontsize=15)
    plt.xlabel('早晨时间', fontsize=15)
    plt.title('清醒数据分布', fontsize=15)
    plt.xticks([-60, -30, 0, 30, 60, 90, 120], wake_labels)
    plt.show()


# --------------------------------------------只有beat参数的逻辑函数--------------------------------------------------------
def test3():
    figsize(10, 6)

    # 只有一个beta参数的逻辑函数
    def logistic(x, beta):
        return 1. / (1. + np.exp(beta * x))

    x = np.linspace(-5, 5, 1000)
    for beta in [-5, -1, 0, 0.5, 1, 5]:
        plt.plot(x, logistic(x, beta), label=r"$\beta$ = %.1f" % beta)

    plt.legend()
    plt.title(r'不同 $\beta$ 参数的逻辑函数')
    plt.show()


# --------------------------------------------不同beat和alpha参数的逻辑函数--------------------------------------------------------
def test4():
    figsize(10, 6)

    def logistic(x, beta, alpha=0):
        return 1.0 / (1.0 + np.exp(np.dot(beta, x) + alpha))

    x = np.linspace(-4, 6, 1000)
    plt.plot(x, logistic(x, beta=1), label=r"$\beta = 1, \alpha = 0$", ls="--", lw=2)
    plt.plot(x, logistic(x, beta=-1), label=r"$\beta = -1, \alpha = 0$", ls="--", lw=2)

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
    plt.title(r'具有不同 $\beta$ 和 $\alpha$ 参数的逻辑函数')
    plt.show()


# --------------------------------------------不同a和t为参数的正太分布曲线--------------------------------------------------------
def test5():
    figsize(10, 6)
    nor = stats.norm
    x = np.linspace(-10, 10, 1000)
    mu = (-5, 0, 4)
    tau = (0.5, 1, 2.5)
    colors = ("forestgreen", "navy", "darkred")

    params = zip(mu, tau, colors)
    for param in params:
        y = nor.pdf(x, loc=param[0], scale=1 / param[1])
        plt.plot(x, y,
                 label="$\mu = %d,\\\tau = %.1f$" % (param[0], param[1]),
                 color=param[2])
        plt.fill_between(x, y, color=param[2], alpha=0.3)

    plt.legend(prop={'size': 18})
    plt.xlabel("$x$")
    plt.ylabel("概率密度", size=18)
    plt.title("正太分布", size=20)
    plt.show()


# --------------------------------------------正太先验变量的参数空间--------------------------------------------------------
def test6():
    figsize(10, 6)
    jet = plt.cm.jet  # 色谱:蓝-青-黄-红
    x = y = np.linspace(0, 5, 100)
    X, Y = np.meshgrid(x, y)
    norm_x = stats.norm.pdf(x, loc=0, scale=1)
    norm_y = stats.norm.pdf(y, loc=0, scale=1)
    M = np.dot(norm_x[:, None], norm_y[None, :])
    im = plt.imshow(M, interpolation='none', origin='lower', cmap=jet)
    plt.xlim(0, 40)
    plt.ylim(0, 40)
    plt.title("探索正太先验的参数空间.")
    plt.show()

    fig2 = plt.figure(2)  # 画第二幅图
    ax = Axes3D(fig2)  # 绘制3D图形
    ax.plot_surface(X, Y, M, cmap=plt.cm.jet)
    ax.view_init(azim=390)
    plt.title("探索正太先验的参数空间.")
    plt.show()


# 逻辑函数
def logistic(x, beta):
    return 1. / (1. + np.exp(beta * x))


def logistic(x, beta, alpha=0):
    return 1.0 / (1.0 + np.exp(np.dot(beta, x) + alpha))


# 以下代码创建模型并执行MCMC，为alpha和beta抽样N_SAMPLES个样本，指定的抽样算法是 Metropolic Hastings ，我们将数据输入模型,并告诉模型数据是伯努利变量，模型于是会为数据找到最有可能的参数alpha和beta

# ---------------------------------睡眠模型以及使用---------------------------------------------------------------------------
def sleep_mode_use():
    # 对时间偏移量进行排序
    sleep_data.sort_values('time_offset', inplace=True)
    # 提取时间偏移量
    time = np.array(sleep_data.loc[:, 'time_offset'])
    # 观察值是indicator
    sleep_obs = np.array(sleep_data.loc[:, 'indicator'])
    with pm.Model() as sleep_model:
        # 创建alpha和beta的先验分布
        alpha = pm.Normal('alpha', mu=0.0, tau=0.01, testval=0.0)
        beta = pm.Normal('beta', mu=0.0, tau=0.01, testval=0.0)
        # 创建一个逻辑函数的确定性变量
        p = pm.Deterministic('p', 1. / (1. + tt.exp(beta * time + alpha)))
        # 创建基于当前数据的伯努利变量
        # pm.Bernoulli('obs', p, observed=sleep_obs)
        # 使用 Metropolis Hastings 抽样
        step = pm.Metropolis()
        # 从后验中抽样
        # 从样本中使用MH采样得到alpha和beta的样本
        # sleep_trace 则保存了模型生成的所有参数值。step 变量指的是特定的算法，

        sleep_trace = pm.sample(N_SAMPLES, step=step)

    # 抽取alpha和beta的样本
    alpha_samples = sleep_trace["alpha"][100:, None]
    beta_samples = sleep_trace["beta"][100:, None]

    figsize(13, 6)
    # -----------------------------------100个样本的alpha和beta的可视化分布-----------------------------------------------------------------
    plt.subplot(211)
    plt.title(r""" %d 个样本的 $\alpha$ 分布""" % N_SAMPLES)

    plt.hist(alpha_samples, histtype='stepfilled',
             color='darkred', bins=30, alpha=0.8, density=True)
    plt.ylabel('概率密度')
    plt.show()

    plt.subplot(212)
    plt.title(r""" %d 个样本的 $\beta$ 分布""" % N_SAMPLES)
    plt.hist(beta_samples, histtype='stepfilled',
             color='darkblue', bins=30, alpha=0.8, density=True)
    plt.ylabel('概率密度')
    plt.show()
    # -----------------------------------5000个样本的睡眠概率分布-----------------------------------------------------------------
    # 设定概率预测的时间长度
    time_est = np.linspace(time.min() - 15, time.max() + 15, int(1e3))[:, None]
    # 取参数的均值
    alpha_est = alpha_samples.mean()
    beta_est = beta_samples.mean()
    # 使用参数的均值所生成的概率
    sleep_est = logistic(time_est, beta_est, alpha_est)

    plt.plot(time_est, sleep_est, color='navy',
             lw=3, label="最有可能的逻辑模型")
    plt.scatter(time, sleep_obs, edgecolor='slateblue',
                s=50, alpha=0.2, label='实际观测值')
    plt.title('%d 个样本的睡眠概率分布' % N_SAMPLES)
    plt.legend(prop={'size': 18})
    plt.ylabel('概率')
    plt.xlabel('下午时间')
    plt.xticks([-60, -30, 0, 30, 60, 90, 120], sleep_labels)
    plt.show()
    print('睡眠概率大于 50% 的时间点位于下午 22:{} '.format(int(time_est[np.where(sleep_est > 0.5)[0][0]][0])))

    colors = ["#348ABD", "#A60628", "#7A68A6"]
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("BMH", colors)
    figsize(12, 6)
    probs = sleep_trace['p']

    plt.scatter(time, probs.mean(axis=0), cmap=cmap,
                c=probs.mean(axis=0), s=50)
    plt.title('睡眠的概率是关于时间的函数')
    plt.xlabel('下午时间')
    plt.ylabel('概率')
    plt.xticks([-60, -30, 0, 30, 60, 90, 120], sleep_labels)

    print('22点的睡眠概率为: {:.2f}%.'.
          format(100 * logistic(0, beta_est, alpha_est)))
    print('21:30的睡眠概率为: {:.2f}%.'.
          format(100 * logistic(-30, beta_est, alpha_est)))
    print('22:30的睡眠概率为: {:.2f}%.'.
          format(100 * logistic(30, beta_est, alpha_est)))

    # ----------------------------------------beta和alpha的置信区间---------------------------------------------------------------------------------
    sleep_all_est = logistic(time_est.T, beta_samples, alpha_samples)
    quantiles = stats.mstats.mquantiles(sleep_all_est, [0.025, 0.975], axis=0)

    plt.fill_between(time_est[:, 0], *quantiles, alpha=0.6,
                     color='slateblue', label='95% 置信区间')
    plt.plot(time_est, sleep_est, lw=2, ls='--',
             color='black', label="睡眠的平均后验概率")
    plt.xticks([-60, -30, 0, 30, 60, 90, 120], sleep_labels)
    plt.scatter(time, sleep_obs, edgecolor='skyblue', s=50, alpha=0.1)
    plt.legend(prop={'size': 14})
    plt.xlabel('PM Time')
    plt.ylabel('Probability')
    plt.title('后验概率的 95% 置信区间')
    plt.show()

    # -----------------------------特定时间的后验概率分布------------------------------------------------------
    def sleep_posterior(time_offset, time):  # 特定时间的后验概率分布
        figsize(16, 8)
        prob = logistic(time_offset, beta_samples, alpha_samples)
        plt.hist(prob, bins=100, histtype='step', lw=4)
        plt.title('  %s点睡眠的概率分布' % time)
        plt.xlabel('睡眠概率')
        plt.ylabel('样本量')
        plt.show()

    sleep_posterior(0, '22:00')
    sleep_posterior(-30, '21:30')

    print('alpha 参数估值: {:.6f}.'.format(alpha_est))
    print('beta 参数估值: {:.6f}.'.format(beta_est))

    # --------------------------------判断马尔可夫链蒙特卡罗模型是否收敛-------------------------------------------------------------------------
    # ------------------轨迹图--------------
    figsize(12, 6)
    plt.subplot(211)
    plt.title(r'Trace of $\alpha$')
    plt.plot(alpha_samples, color='darkred')
    plt.xlabel('样本量')
    plt.ylabel('参数')
    plt.show()

    plt.subplot(212)
    plt.title(r'Trace of $\beta$')
    plt.plot(beta_samples, color='b')
    plt.xlabel('样本量')
    plt.ylabel('参数')
    plt.tight_layout(h_pad=0.8)
    plt.show()


# ----------------------清醒数据建模及使用------------------------------------------------------------------------
def wake_mode_use():
    wake_data.sort_values('time_offset', inplace=True)
    time = np.array(wake_data.loc[:, 'time_offset'])
    wake_obs = np.array(wake_data.loc[:, 'indicator'])
    # ----------------------------------清醒数据模型---------------------------------
    with pm.Model() as wake_model:
        alpha = pm.Normal('alpha', mu=0.0, tau=0.01, testval=0.0)
        beta = pm.Normal('beta', mu=0.0, tau=0.01, testval=0.0)
        p = pm.Deterministic('p', 1. / (1. + tt.exp(beta * time + alpha)))
        # observed = pm.Bernoulli('obs', p, observed=wake_obs)
        step = pm.Metropolis()
        wake_trace = pm.sample(N_SAMPLES, step=step)
    # ------------------------------100个样本的后验概率--------------------------------------------------------------------
    alpha_samples = wake_trace["alpha"][100:, None]
    beta_samples = wake_trace["beta"][100:, None]
    time_est = np.linspace(time.min() - 15, time.max() + 15, int(1e3))[:, None]
    alpha_est = alpha_samples.mean()
    beta_est = beta_samples.mean()
    wake_est = logistic(time_est, beta=beta_est, alpha=alpha_est)

    figsize(13, 6)
    plt.plot(time_est, wake_est, color='darkred',
             lw=3, label="清醒时候的平均睡眠后验概率")
    plt.scatter(time, wake_obs, edgecolor='r', facecolor='r',
                s=50, alpha=0.05, label='观测值')
    plt.title('%d个样本的后验概率' % N_SAMPLES)
    plt.legend(prop={'size': 14})
    plt.ylabel('概率')
    plt.xlabel('上午时间')
    plt.xticks([-60, -30, 0, 30, 60, 90, 120], wake_labels)
    plt.show()

    print('清醒的概率大于50%的时间点位于上午 6:{}'.format(int(time_est[np.where(wake_est < 0.5)][0])))

    colors = ["#348ABD", "#A60628", "#7A68A6"]
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("BMH", colors)
    figsize(12, 6)
    probs = wake_trace['p']
    # ------------------------------上午时间的睡眠概率--------------------------------------------------------------------
    plt.scatter(time, probs.mean(axis=0), cmap=cmap,
                c=probs.mean(axis=0), s=50)
    plt.title('上午时间的睡眠概率')
    plt.xlabel('上午时间')
    plt.ylabel('概率')
    plt.xticks([-60, -30, 0, 30, 60, 90, 120], wake_labels)
    plt.show()

    print('上午5:30 清醒的概率: {:.2f}%.'.
          format(100 - (100 * logistic(-30, beta=beta_est, alpha=alpha_est))))
    print('上午6:00清醒的概率: {:.2f}%.'.
          format(100 - (100 * logistic(0, beta=beta_est, alpha=alpha_est))))
    print('上午6:30清醒的概率: {:.2f}%.'.
          format(100 - (100 * logistic(30, beta=beta_est, alpha=alpha_est))))


# -------------------------------睡眠时间长度模型及使用-------------------------------------------------------------------------
def sleep_time_mode_use():
    raw_data = pd.read_csv(
        'D:/weChatFile/WeChat Files/wxid_fg4c7ci7wpud21/FileStorage/File/2021-04/sleep_wake.csv')
    raw_data['length'] = 8 - (raw_data['Sleep'] / 60) + (raw_data['Wake'] / 60)
    duration = raw_data['length']
    # -----------------------------睡眠时间长度-------------------------------------------------------------
    figsize(10, 8)
    plt.hist(duration, bins=20, color='darkred')
    plt.xlabel('小时')
    plt.title('睡眠时间长度分布')
    plt.ylabel('观测值')
    plt.show()
    # ---------------------------右偏睡眠时间长度概率密度----------------------------------------
    a = 3
    fig, ax = plt.subplots(1, 1)
    x = np.linspace(6, 12, int(1e3))

    figsize(10, 8)
    plt.hist(duration, bins=20, color='darkred', density=1, stacked=True)
    plt.xlabel('小时')
    plt.title('右偏的睡眠时间长度的概率密度(PDF)')
    plt.ylabel('观测值')
    plt.plot(x, stats.skewnorm.pdf(x, a, loc=7.4, scale=1), 'r-',
             lw=3, label='skewnorm pdf')
    plt.show()

    # ------------------------------睡眠长度概率模型--------------------------------------------------
    with pm.Model() as duration_model:
        # 定义三个参数的先验概率分布其中我们增加了一个偏度参数alpha_skew
        alpha_skew = pm.Normal('alpha_skew', mu=0, tau=0.5, testval=3.0)
        mu_ = pm.Normal('mu', mu=0, tau=0.5, testval=7.4)
        tau_ = pm.Normal('tau', mu=0, tau=0.5, testval=1.0)

        # Duration 为一个确定性变量
        duration_ = pm.SkewNormal('duration', alpha=alpha_skew, mu=mu_,
                                  sd=1 / tau_, observed=duration)

        # Metropolis Hastings 抽样
        step = pm.Metropolis()
        duration_trace = pm.sample(N_SAMPLES, step=step)
    # --------------------抽取最有可能的估值参数---------------------------------------------------------
    # 抽取最有可能的估值参数
    alpha_skew_samples = duration_trace['alpha_skew'][1000:]
    mu_samples = duration_trace['mu'][1000:]
    tau_samples = duration_trace['tau'][1000:]

    alpha_skew_est = alpha_skew_samples.mean()
    mu_est = mu_samples.mean()
    tau_est = tau_samples.mean()
    # -----------------------睡眠长度后验分布长度可视化-------------------------------------------------------
    x = np.linspace(6, 12, 1000)
    y = stats.skewnorm.pdf(x, a=alpha_skew_est, loc=mu_est, scale=1 / tau_est)
    plt.plot(x, y, color='forestgreen')
    plt.fill_between(x, y, color='forestgreen', alpha=0.2)
    plt.xlabel('小时')
    plt.ylabel('概率')
    plt.title('睡眠时间长度的后验分布')
    plt.vlines(x=x[np.argmax(y)], ymin=0, ymax=y.max(),
               linestyles='--', linewidth=2, color='red',
               label='最可能的睡眠时间长度')
    plt.show()
    print('最可能的睡眠时间长度为 {:.2f} 小时.'.format(x[np.argmax(y)]))
    # -----------------------查询后验概率模型--------------------------------------------------------------
    print('睡眠时间至少6.5小时的概率为:{:.2f}%.'.
          format(100 * (1 - stats.skewnorm.cdf(6.5, a=alpha_skew_est, loc=mu_est, scale=1 / tau_est))))
    print('睡眠时间至少8小时的概率为:{:.2f}%.'.
          format(100 * (1 - stats.skewnorm.cdf(8.0, a=alpha_skew_est, loc=mu_est, scale=1 / tau_est))))
    print('睡眠时间至少9小时的概率为:{:.2f}%.'.
          format(100 * (1 - stats.skewnorm.cdf(9.0, a=alpha_skew_est, loc=mu_est, scale=1 / tau_est))))
    # -------------------------可视化后验和数据-------------------------------------------------------------------------------------
    x = np.linspace(6, 12, 1000)
    y = stats.skewnorm.pdf(x, a=alpha_skew_est, loc=mu_est, scale=1 / tau_est)
    figsize(10, 8)
    # 绘制后验概率分布
    plt.plot(x, y, color='forestgreen',
             label='Model', lw=3)
    plt.fill_between(x, y, color='forestgreen', alpha=0.2)

    # 绘制观测值直方图
    plt.hist(duration, bins=10, color='red', alpha=0.8,
             label='观测值', density=1, stacked=True)
    plt.xlabel('小时')
    plt.ylabel('概率')
    plt.title('模型')
    plt.vlines(x=x[np.argmax(y)], ymin=0, ymax=y.max(),
               linestyles='--', linewidth=2, color='k',
               label='最可能的睡眠时间长度')
    plt.legend(prop={'size': 12})
    plt.show()


# 主函数，进行调用
if __name__ == '__main__':
    sleep_mode_use() #调用睡眠模型
    # wake_mode_use()  #调用清醒数据模型
    # sleep_time_mode_use() #调用睡眠长度模型
    # test1()  # 睡眠数据分布_散点图
    # test2()  # 清醒数据分布_散点图
    # test3()  # 只有beat参数的逻辑函数
    # test4()  # 不同beat和alpha参数的逻辑函数
    # test5()  # 不同a和t为参数的正太分布曲线
    # test6()  # 正太先验变量的参数空间
