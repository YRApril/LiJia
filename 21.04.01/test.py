# encoding: utf-8
import pandas as pd

path = 'data/train13519-1004-1123-1124-new.csv'
time = '2015-10-04'

# print(path)
df = pd.read_csv(path)

print(df['timestamp'])
print(df['timestamp'][0])
print(type(df['timestamp'][0]))

print(df['timestamp'][0].lstrip().rstrip().split(' ')[0])

df['timeIndex'] = df['timestamp'].apply(lambda x: x.lstrip().rstrip().split(' ')[0])

print(df)


data = df.query('timeIndex == \'' + time + '\'')

print(data)

# data.drop('timeIndex', axis=1, inplace=True)
data2 = data.drop('timeIndex', axis=1)

print(data2)
