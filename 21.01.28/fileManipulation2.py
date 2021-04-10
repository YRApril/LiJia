# 导入相关模块
import numpy as np
import pandas as pd
import time

fileName = 'data/train13519-ys.csv'


def copeData(fileName):
    # data = pd.read_csv('F:/machineLearning/DateSet/hangzhou-mougaojiadaolu/archive/DataSet/train13519-jg-20151004.csv', header=0,usecols=[0,1,2])
    data = pd.read_csv(fileName, header=0, usecols=[0, 1])
    data['timestamp'] = pd.to_datetime(data['timestamp'])  # 将数据类型转化为日期类型
    data = data.set_index('timestamp')  # 将timestamp设置为索引
    # #处理行
    # print(data)
    # print(data['2015-10-04'])   #获取某日数据
    # print(data['2015-10'])
    # print(data['2015-10'] + data['2015-11'])  # 获取某月数据
    data2 = pd.concat([data['2015-10'], data['2015-11']])
    data2.to_csv('data/aaa.csv')
    print(data2)
    return data2

# trainx,trainy=copeData(fileName)
# print(trainx)
# print(trainy)
xxx = copeData(fileName)
print(xxx)
