import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from sklearn import preprocessing
from tensorflow.python.keras.layers import LSTM,Dense,Dropout
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


#绘制混淆矩阵函数
def plot_confusion_matrix(cm,labels, title='Confusion Matrix', cmap=plt.cm.binary):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    xlocations = np.array(range(len(labels)))
    plt.xticks(xlocations, labels, rotation=90)
    plt.yticks(xlocations, labels)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig('LSTM.pdf')
    plt.show()


data = pd.read_csv('F:/machineLearning/DateSet/DataAnalysisTask/stock/stock.csv',header=0)
datay = data['label']
datax = data[data.columns[0:7]]

#标准化
datax = preprocessing.scale(datax)

#每七天数据拼接为一个向量
x = []
y = []
for i in range(0,len(datax),7):
	if i+7<=len(datay)-1:
		y.append(datay[i+6])
		x.append(np.array(datax[i:i+7]).flatten().reshape(7,7))

#设置成numpy数组
x = np.concatenate([x[0:-1]],axis=1)
y = np.concatenate([y[0:-1]],axis=0)

print('x',x.shape)
print('y',y.shape)


print('---------------数据分布情况---------------')
print('0：',len(np.where(y==0)[0]))
print('1：',len(np.where(y==1)[0]))
print('-1：',len(np.where(y==-1)[0]))


#删除标签0的数据
x = np.delete(x,np.where(y==0)[0],axis=0)
y = np.delete(y,np.where(y==0)[0])
#设置-1标签为0
y[np.where(y==-1)[0]] = 0


print('---------------数据维度---------------')
print(x.shape)
print(y.shape)




#每组采样300条数据用于训练
index0 = np.random.permutation(np.where(y==0)[0])
index1 = np.random.permutation(np.where(y==1)[0])
 

 

#准备训练数据
x_train = np.concatenate((x[index0[0:300]],x[index1[0:300]]), axis=0)
y_train = np.concatenate((y[index0[0:300]],y[index1[0:300]]), axis=0)


#准备测试数据
x_test = np.concatenate((x[index0[300:400]],x[index1[300:400]]), axis=0)
y_test = np.concatenate((y[index0[300:400]],y[index1[300:400]]), axis=0)





#定义神经网络
model = tf.keras.Sequential()
model.add(LSTM(50, input_shape=(x_train.shape[1], x_train.shape[2])))
model.add(Dense(1024, activation='relu'))   
model.add(Dropout(0.5))
model.add(Dense(128, activation='relu'))   
model.add(Dropout(0.25))
model.add(Dense(1,activation='sigmoid')) 

# 输出层
model.summary()
 
# 定义优化器
sgd = tf.keras.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
 
# 整合模型
model.compile(loss='binary_crossentropy',
             optimizer=sgd,
             metrics=['accuracy'])

#开始训练
pre = np.random.permutation(x_train.shape[0])
history = model.fit(x_train[pre],y_train[pre],epochs=600)
 
# history.history.key()  # ['loss', 'acc']
 
plt.plot(history.epoch, history.history.get('loss'),'bo',label='loss')
plt.plot(history.epoch, history.history.get('acc'),'g',label='acc')
plt.xlabel('Epochs')
plt.ylabel('loss+acc')
plt.legend(loc='best')  # 图例
plt.savefig('LSTM_train.pdf')
plt.show()


#预测测试集
test = model.predict_classes(x_test)


#打印结果
report = classification_report(y_test, test)
print(report)
#绘制混淆矩阵
matrix = confusion_matrix(y_test, test)
plot_confusion_matrix(matrix,[0,1])