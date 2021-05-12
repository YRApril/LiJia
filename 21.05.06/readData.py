import pandas as pd
from sklearn.manifold import TSNE
from sklearn import preprocessing
import numpy as np


users = []
products = []

fileName = "data/score.txt"
replaceNaNAsZero = True


def readDataAsDataFrame():
    file = open(fileName, "r")
    for line in file.readlines():
        line = line.strip()
        user = int(line.split(',')[0])
        product = int(line.split(',')[1])
        if not user in users:
            users.append(user)
        if not product in products:
            products.append(product)
    # 关闭文件
    file.close()
    users.sort()
    products.sort()
    data = pd.DataFrame(index=products, columns=users)
    file = open(fileName, "r")
    for line in file.readlines():
        line = line.strip()
        user = int(line.split(',')[0])
        product = int(line.split(',')[1])
        sorce = float(line.split(',')[2])
        data[user][product] = sorce
    # 关闭文件
    file.close()
    data = data.fillna(data.mean())
    data = pd.DataFrame(data.values.T, index=data.columns, columns=data.index)  # 转置
    return data


def get2DimensionValue(data):
    """
    降维
    :param data:  原始数据 dataframe
    :return: 降维后的ndarray
    """
    dataArray = np.array(data)
    tDimensionValue = preprocessing.scale(dataArray) #标准化
    tsne = TSNE()
    temp_trans = tsne.fit_transform(tDimensionValue)
    # print(temp_trans)
    return temp_trans


#
# data = readDataAsDataFrame()
# data = get2DimensionValue(data)
# print(data)
# print(data.shape)
