# encoding: utf-8
import pandas as pd

path = 'data/train13519-1004-1123-1124-new.csv'
timePre = '2015-10-'

# print(path)
df = pd.read_csv(path)

data = pd.DataFrame(columns=['num1', 'num', 'timestamp', 'hourly_traffic_count'])

# print(data)


def getDataInDate(date):
    if date < 10:
        date = '0' + str(date)
    else:
        date = str(date)
    df["timeIndex"] = df['timestamp'].apply(lambda x: x.lstrip().rstrip().split(' ')[0])
    date = df.query('timeIndex == \'' + timePre + date + '\'')
    data2 = date.drop('timeIndex', axis=1)
    # print(data2)
    return data2


for i in range(4, 15):
    # print(i)
    mData = getDataInDate(i)
    # print(mData)
    data = data.append(mData)

print(data)
