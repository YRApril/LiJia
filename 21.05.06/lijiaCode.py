import pandas as pd
from sklearn.cluster import KMeans #导入K均值聚类算法
import numpy as np
from pandas import plotting
import matplotlib.pyplot as plt  #matplotlib画图
import seaborn as sns
import readData

# data = pd.read_csv('F:/machineLearning/DateSet/DataAnalysisTask/Kmeans/score.csv')
# data = readData.readDataAsDataFrame()
data = pd.read_csv('data/data111.csv')
df=pd.DataFrame(data)
print(df)

#数据集分布情况
sns.set(palette="muted", color_codes=True)  # seaborn样式
plt.rcParams['axes.unicode_minus'] = False  # 解决无法显示符号的问题
sns.set(font='SimHei', font_scale=0.8)    # 解决Seaborn中文显示问题
# 绘图
plt.figure(1, figsize=(13, 6))
n = 0
for x in ['userId', 'productId', 'productScore']:
  n += 1
  plt.subplot(1, 3, n)
  plt.subplots_adjust(hspace=0.5, wspace=0.5)
  sns.distplot(df[x], bins=16, kde=True)  # kde 密度曲线
  plt.title('{}分布情况'.format(x))
  plt.tight_layout()
plt.show()