import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import test1


def calcDis(dataSet, centroids, k):
    """
    计算各点到当前质心的距离
    :param dataSet: 各点的数据集
    :param centroids: 当前的质心
    :param k: 质心数量
    :return: 返回各点到质心距离列表，第一列为第一个质心的距离，第二列为第二个质心的距离
    """
    clalist = []  # 节点到质心距离列表，第一列为第一个质心的距离，第二列为第二个质心的距离
    print('-------------------start---------------------------')
    for data in dataSet:
        print('打印data', data)
        data1 = np.tile(data, (k, 1))  # 经过复制得到一个k行1列的array，成员为list
        print('在y轴上复制')
        print(data1)
        print('打印centroids', centroids)
        diff = np.tile(data, (k, 1)) - centroids  # 当前点 x，y 分别与 两个质心x，y 的差值
        # 相减   (np.tile(a,(2,1))就是把a先沿x轴复制1倍，即没有复制，仍然是 [0,1,2]。
        # 再把结果沿y方向复制2倍得到array([[0,1,2],[0,1,2]]))
        print('diff', diff)
        squaredDiff = diff ** 2  # 平方
        print('squaredDiff', squaredDiff)
        squaredDist = np.sum(squaredDiff, axis=1)  # 和  (axis=1表示行)
        print('squaredDist', squaredDist)
        distance = squaredDist ** 0.5  # 开根号 当前点分别到两个质心的距离
        print('distance', distance)
        clalist.append(distance)  # 距离添加到节点质心距离列表
        print('clalist', clalist)
    clalist = np.array(clalist)  # 返回一个每个点到质点的距离len(dateSet)*k的数组
    print('1111clalist', clalist)
    return clalist


def classify(dataSet, centroids, k):
    """
    计算质心
    :param dataSet: 各点的坐标信息
    :param centroids: 当前质心位置
    :param k: 质心的数量
    :return:返回质心变化量，新的质心
    """
    # TODO 计算样本到质心的距离
    clalist = calcDis(dataSet, centroids, k)  # 第一列为点到第一个质心的距离，第二列为点到第二个质心的距离
    print("样本到质心的距离:")
    print(clalist)
    print()

    # TODO 分组并计算新的质心
    minDistIndices = np.argmin(clalist, axis=1)  # axis=1 表示求出每行的最小值的下标（每个点和哪个质心最接近）
    print('minDistIndices')
    print(minDistIndices)

    newCentroids = pd.DataFrame(dataSet).groupby(minDistIndices).mean()  # 根据相近的质心分组，并计算xy轴均值作为新的质心
    print('newCentroids')
    print(newCentroids)
    # DataFramte(dataSet)对DataSet分组，groupby(min)按照min进行统计分类，mean()对分类结果求均值
    newCentroids = newCentroids.values

    # 计算变化量
    changed = newCentroids - centroids

    return changed, newCentroids


def kmeans(dataSet, k):
    """
    使用k-means对数据集进行分类
    :param dataSet: 各点数据集（坐标）
    :param k: 质心数量/分类数量
    :return: 返回质心列表，分簇结果
    """
    # 随机取质心
    centroids = random.sample(dataSet, k)  # 从数据集中随机取质心

    # print("打印centroids",centroids)

    # 更新质心 直到变化量全为0
    changed, newCentroids = classify(dataSet, centroids, k)  # 计算质心，判断变化量
    while np.any(changed != 0):  # 变化量不为0，继续更新质心
        changed, newCentroids = classify(dataSet, newCentroids, k)  # 计算质心，判断变化量

    centroids = sorted(newCentroids.tolist())  # tolist()将矩阵转换成列表 sorted()排序

    # 根据质心计算每个集群
    cluster = []
    clalist = calcDis(dataSet, centroids, k)  # 计算各点与最终质心的距离
    minDistIndices = np.argmin(clalist, axis=1)  # 求出每行的最小值的下标（每个点和哪个质心最接近）
    print(minDistIndices)
    for i in range(k):
        cluster.append([])
    for i, j in enumerate(minDistIndices):  # enymerate()可同时遍历索引和遍历元素
        cluster[j].append(dataSet[i])

    return centroids, cluster


def createDataSet():
    """
    创建数据集（各点的位置）
    :return: 返回包含各点位置的数据集
    """
    return [[1, 1], [1, 2], [2, 1], [6, 4], [6, 3], [5, 4]]


if __name__ == '__main__':
    # dataset = createDataSet()  # 数据集
    dataset = test1.getData()  # 数据集
    print('数据集：', dataset)
    exit()
    centroids, cluster = kmeans(dataset, 2)
    print('质心为：%s' % centroids)
    print('集群为：%s' % cluster)

    for i in range(len(dataset)):
        plt.scatter(dataset[i][0], dataset[i][1], marker='o', color='green', s=40, label='原始点')
        #  记号形状       颜色      点的大小      设置标签
        for j in range(len(centroids)):
            plt.scatter(centroids[j][0], centroids[j][1], marker='x', color='red', s=50, label='质心')
            plt.show()
