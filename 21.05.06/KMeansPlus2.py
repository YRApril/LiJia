import numpy as np
import random
from sklearn.cluster import KMeans
from task2.task2_execute import readData

# 测试执行
# X = np.array([[0, 2], [0, 0], [1, 0], [5, 0], [5, 2]])
X = readData.get2DimensionValue(readData.readDataAsDataFrame())
# 模型创建
kmeans_model=KMeans(n_clusters=5,init='k-means++',random_state=11)
# 进行聚类处理
y_pred=kmeans_model.fit_predict(X)
print("y_pred", y_pred)
