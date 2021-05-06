import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn import datasets
import pandas as pd

inputfile = 'F:/machineLearning/DateSet/Kmeans/data111.csv'
np.set_printoptions(suppress=True)
data = pd.read_csv(inputfile)
#print(data)
dataset = data.values
X = dataset[:,:3]    #取第一列第二列
print(X)

# 绘制数据分布图
plt.scatter(X[:, 1], X[:, 2], c="red", marker='o', label='see')
plt.xlabel('good number')
plt.ylabel('good scord')
plt.legend(loc=2)
plt.show()

estimator = KMeans(n_clusters=3)   # 构造聚类器
estimator.fit(X) # 聚类
label_pred = estimator.labels_    # 获取聚类标签
print(label_pred)
print(len(label_pred))
# 绘制k-means结果
x0 = X[label_pred == 0]
x1 = X[label_pred == 1]
x2 = X[label_pred == 2]
plt.scatter(x0[:, 1], x0[:, 2], c="red", marker='o', label='label0')
plt.scatter(x1[:, 1], x1[:, 2], c="green", marker='*', label='label1')
plt.scatter(x2[:, 1], x2[:, 2], c="blue", marker='+', label='label2')
plt.xlabel('sepal length')
plt.ylabel('sepal width')
plt.legend(loc=2)
plt.show()