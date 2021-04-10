import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def getData():
    data = pd.read_csv('data/testSet.csv', header=0, usecols=[1, 2])
    data = np.array(data).tolist()
    print(data)
    return data


# 定义读取文件的函数
getData()