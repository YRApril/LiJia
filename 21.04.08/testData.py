# encoding: utf-8


import pandas as pd
import time


def getData():
    list2 = ['温度', '湿度', '风速1', '风向1', '风速2', '风向2', '总辐射1', '时总辐射1', '总辐射2', '时总辐射2', '直辐射', '时直辐射', '反辐射',
             '时反辐射', '散辐射', '时散辐射']

    list = []
    for i in range(0, 16):
        list.append(i)

    timeStart = time.time()
    data = pd.read_excel('data/2018QXZ.xlsx')

    data.drop(index=[0], columns=['气象站数据记录', 'Unnamed: 1', 'Unnamed: 2'], inplace=True)

    data.columns = list2
    print(data)
    print('读取数据用时：' + str(round((time.time() - timeStart) / 60, 2)) + 'min\n')
    return data



