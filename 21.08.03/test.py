import pandas as pd
import datetime
import matplotlib.pyplot as plt

# input_file = '../5.20210802/data/train13519-1004-1123.csv'
input_file = 'data/train13519-1004-1123-1124-new.csv'
df = pd.read_csv(input_file, header=0)
df['week'] = ''
# print(df)
# print(df.iloc[0]['timestamp'])
# print(type(df.iloc[0]['timestamp']))
# print(df.iloc[0]['timestamp'].strip().split(' ')[0])

df['timestamp'] = df['timestamp'].apply(lambda x: x.strip().split(' ')[0])

# print(df.iloc[0]['timestamp'].strip())
# print(df.iloc[0]['timestamp'].strip().split(' ')[0].split('-')[0])
# print(df.iloc[0]['timestamp'].strip().split(' ')[0].split('-')[1])
# print(int(df.iloc[0]['timestamp'].strip().split(' ')[0].split('-')[2]))
# day=datetime.datetime(2020,7,27).strftime("%w")
# day = datetime.datetime(2015, 10, 4).strftime("%w")
# print(day)

df['week'] = df['timestamp'].apply(lambda x: datetime.datetime(
    int(x.strip().split(' ')[0].split('-')[0]),
    int(x.strip().split(' ')[0].split('-')[1]),
    int(x.strip().split(' ')[0].split('-')[2])
).strftime("%w"))

print(df)

df.to_csv('data/train13519-1004-1123-1124-new-new.csv')
