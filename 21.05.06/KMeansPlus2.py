import numpy as np
import random
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix
import readData
import givenResultProduce
from matplotlib import pyplot as plt

# 测试执行
# X = np.array([[0, 2], [0, 0], [1, 0], [5, 0], [5, 2]])
X = readData.get2DimensionValue(readData.readDataAsDataFrame())

# 模型创建
kmeans_model = KMeans(n_clusters=5, init='k-means++', random_state=11)
# 进行聚类处理
y_pred = kmeans_model.fit_predict(X)
# print("y_pred", y_pred)



lists = givenResultProduce.fullArrangement()

for list in lists:
    print()
    print(list)
    y_true = np.array(givenResultProduce.getGroupResult(list))
    y_true = y_true.flatten()
    # print("y_true", y_true)

    cm = confusion_matrix(y_true.astype(int), y_pred.astype(int))
    conf_matrix = pd.DataFrame(cm, index=["0", "1", "2", "3", "4"], columns=["0", "1", "2", "3", "4"])
    print(conf_matrix)
    print()

    plt.matshow(conf_matrix, cmap=plt.gray())
    plt.title(str(list))
    plt.savefig("out/" + str(list) + ".png")
    plt.show()
