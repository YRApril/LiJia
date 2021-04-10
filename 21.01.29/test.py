# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

# fileName = 'data/oldData.csv'
fileName = 'data/train13519-1004-1123-1124.csv'

data = pd.read_csv(fileName, names=['timestamp', 'hourly_traffic_count'])
newData = pd.DataFrame(columns=['num', 'timestamp', 'hourly_traffic_count'])
# print(data)

# todo 获取日期列
timestamps = data['timestamp']
# print(timestamps)
# print(type(timestamps))
# print(type(timestamps[1]))

# todo 只保留到日
timestamps = timestamps.apply(lambda x: x[:-8].rstrip())
# print(timestamps)

# todo 去除重复日期
times = timestamps.drop_duplicates().tolist()
# print(times)

# todo 划分dataframe 编号 合并
for time in times:
    sData = data[data['timestamp'].str.contains(time)]
    sData.insert(0, 'num', np.arange(0, sData.shape[0], 1).tolist())
    newData = newData.append(sData)


print(newData)
newData.to_csv('data/new.csv')

