# 导入相关模块
import datetime

import numpy as np
import time
import pandas as pd
#timestamp,hourly_traffic_count
fileName = 'data/train13519-1004-1123-1124.csv'


def time_increase(begin_time, days):
    ts = time.strptime(str(begin_time), "%Y-%m-%d")
    ts = time.mktime(ts)
    dateArray = datetime.datetime.utcfromtimestamp(ts)
    date_increase = (dateArray + datetime.timedelta(days=days)).strftime("%Y-%m-%d")
    # print("日期：{}".format(date_increase))
    return date_increase


# aa = time_increase('2015-10-04', 2)
# print(aa)

def copeData(fileName):
    lines = []
    x = np.arange(1, 289, 1)
    i = 0
    begin_time = '2015-10-04'
    # print('begin_time', begin_time)
    with open(fileName, 'r') as file_to_read:
        while True:
            line = file_to_read.readline()
            if not line:
                break
            line = line.strip('\n')
            day=line.split(' ')[0]
            #print(day)
            if(day==begin_time):
                #print(x[i])
                #line=np.array(line)
                #print(line)
                line=np.hstack([line, x[i]])
                # print(line)
                i+=1
            else:
                begin_time=time_increase(begin_time,2)
                #print(begin_time)
                i=0
            lines.append(line)
            #print('lines:\n',lines)
        return lines

lineslist=copeData(fileName)
lineslist=np.array(lineslist)
# print(lineslist)
