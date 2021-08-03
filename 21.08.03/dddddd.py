import pandas as pd
import datetime
import matplotlib.pyplot as plt


# input_file = '../5.20210802/data/train13519-1004-1123.csv'
input_file = 'data/train13519-1004-1123-1124-new.csv'
df = pd.read_csv(input_file, header=0)
df

def change(time):
    try:
        time=datetime.datetime.strptime(time.strip(),"%Y-%m-%d %H:%M:%m")
    except:
        time=datetime.datetime.strptime(time.strip(),"%Y/%m/%d %H:%M:%m")
    return datetime.datetime.strftime(time,"%Y-%m-%d %H:%M:%m")
df['timestamp']=df['timestamp'].apply(change)
df['timestamp']

def change2(time):
    size=len(time)
    return time[-5:size]
df['time']=df['timestamp'].apply(change2)

df2=df.groupby('time')

for index,data in df2:
    if index=='00:20':
        X=data.iloc[:,2]
        y=data.iloc[:,3]
        print(data.iloc[:,2:4])
        print("x\n",X)
        print("y\n",y)
        
    def printResult(x,y):
        day=x.split(' ',1)
        print("day",day)
#         plt.ion()
#         plt.plot(x, y, 'r', label='every day trafficFlow')
#         fig = plt.gcf()
#         fig.set_size_inches(18.5, 10.5)
#         plt.xlabel("day")
#         plt.ylabel("trafficFlow")
#         plt.title("trafficFlow")
#         plt.legend()
#         plt.savefig('../5.20210802/figure/my_fig2.png')
#         plt.show()
#         plt.close()


printResult(X,y)