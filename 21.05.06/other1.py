import readData
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist

data = readData.readDataAsDataFrame()

data2 = readData.get2DimensionValue(data)
print(data2)




def elbow_rule(data):
    # 肘部法则 求解最佳分类数
    # K-Means参数的最优解也是以成本函数最小化为目标
    # 成本函数是各个类畸变程度（distortions）之和。每个类的畸变程度等于该类重心与其内部成员位置距离的平方和
    a = []
    x = []
    y = []
    for k in range(1, 10):
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(data)
        value = sum(np.min(cdist(data, kmeans.cluster_centers_, 'euclidean'), axis=1)) / data.shape[0]
        print(k, value)
        x.append(k)
        y.append(value)
        a.append(value)

    cha = [a[i] - a[i + 1] for i in range(len(a) - 1)]
    a_v = a[cha.index(max(cha)) + 1]
    index = a.index(a_v) + 1
    print()
    print(max(cha), a_v,index)

    return index, x, y


def getK():
    index, x, y = elbow_rule(data2)
    plt.xlabel('k')
    plt.ylabel('SSE')
    plt.plot(x, y, 'o-')
    plt.show()







getK()
