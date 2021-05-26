#coding=gbk
'''
Created on 2017年2月20日
@author: Lu.yipiao
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import warnings

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
warnings.filterwarnings('ignore')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
#定义常量
rnn_unit=10       #hidden layer units
input_size=6     #输入神经元个数
output_size=1    #输出神经元个数
lr=0.0006         #学习率
#——————————————————导入数据——————————————————————
#df=pd.read_csv("E:\\pythonProject-tensorflow\\股票数据.csv")
df=pd.read_csv("data/股票数据.csv")
# df=pd.read_csv("F:/machineLearning/DateSet/DataAnalysisTask/stock/股票数据.csv")
data=df.iloc[:,1:8].values #取第2-8列,data看似一个6529*7的一个矩阵
#df.shape=(6529,16)
#获取训练集
def get_train_data(batch_size=60,time_step=20,train_begin=0,train_end=5000):
    batch_index=[]
    data_train=data[train_begin:train_end]
    #标准化也叫做归一化，一般是将数据映射到指定的范围，用于除去不同维度数据的量纲以及量纲单位
    normalized_train_data=(data_train-np.mean(data_train,axis=0))/np.std(data_train,axis=0)  #标准化
    train_x,train_y=[],[]   #训练集
    for i in range(len(normalized_train_data)-time_step):
       #banch_size为每批次训练样本数
       if i % batch_size==0:
           batch_index.append(i)
       x=normalized_train_data[i:i+time_step,:6]#前6列为输入维度数据
       y=normalized_train_data[i:i+time_step,6,np.newaxis]
       train_x.append(x.tolist())
       train_y.append(y.tolist())
    batch_index.append((len(normalized_train_data)-time_step))
    return batch_index,train_x,train_y
#获取测试集
def get_test_data(time_step=20,test_begin=5000):
    data_test=data[test_begin:]
    mean=np.mean(data_test,axis=0)
    std=np.std(data_test,axis=0)
    normalized_test_data=(data_test-mean)/std  #标准化
    size=(len(normalized_test_data)+time_step-1)//time_step  #有size个sample
    test_x,test_y=[],[]
    for i in range(size-1):
       x=normalized_test_data[i*time_step:(i+1)*time_step,:6]
       y=normalized_test_data[i*time_step:(i+1)*time_step,6]
       test_x.append(x.tolist())
       test_y.extend(y)
    test_x.append((normalized_test_data[(i+1)*time_step:,:6]).tolist())
    test_y.extend((normalized_test_data[(i+1)*time_step:,6]).tolist())
    return mean,std,test_x,test_y
#——————————————————定义神经网络变量——————————————————
#输入层、输出层权重、偏置
weights={
         'in':tf.Variable(tf.random.normal([input_size,rnn_unit])),
         'out':tf.Variable(tf.random.normal([rnn_unit,1]))
        }
biases={
        'in':tf.Variable(tf.constant(0.1,shape=[rnn_unit,])),
        'out':tf.Variable(tf.constant(0.1,shape=[1,]))
       }
#——————————————————定义神经网络变量——————————————————
def lstm(X):
    batch_size=tf.shape(X)[0]
    time_step=tf.shape(X)[1]
    w_in=weights['in']
    b_in=biases['in']
    input=tf.reshape(X,[-1,input_size])  #需要将tensor转成2维进行计算，计算后的结果作为隐藏层的输入
    input_rnn=tf.matmul(input,w_in)+b_in
    input_rnn=tf.reshape(input_rnn,[-1,time_step,rnn_unit])  #将tensor转成3维，作为lstm cell的输入
    cell=tf.nn.rnn_cell.BasicLSTMCell(rnn_unit)
    init_state=cell.zero_state(batch_size,dtype=tf.float32)
    output_rnn,final_states=tf.nn.dynamic_rnn(cell, input_rnn,initial_state=init_state, dtype=tf.float32)  #output_rnn是记录lstm每个输出节点的结果，final_states是最后一个cell的结果
    output=tf.reshape(output_rnn,[-1,rnn_unit]) #作为输出层的输入
    w_out=weights['out']
    b_out=biases['out']
    pred=tf.matmul(output,w_out)+b_out
    return pred,final_states
#——————————————————训练模型——————————————————
# batch_sizie为每批次训练样本数，time_step为时间步,后面两个参数决定训练集的数量
def train_lstm(batch_size=60,time_step=20,train_begin=0,train_end=5000):
    #tf.placeholder()函数作为一种占位符用于定义过程，可以理解为形参，在执行的时候再赋具体的值。
    #不必指定初始值，可在运行时，通过 Session.run 的函数的 feed_dict 参数指定。
    X=tf.placeholder(tf.float32, shape=[None,time_step,input_size])
    Y=tf.placeholder(tf.float32, shape=[None,time_step,output_size])
    batch_index,train_x,train_y=get_train_data(batch_size,time_step,train_begin,train_end)
    #定义神经网络变量
    pred,_=lstm(X)
    #损失函数
    loss=tf.reduce_mean(tf.square(tf.reshape(pred,[-1])-tf.reshape(Y, [-1])))
    train_op=tf.train.AdamOptimizer(lr).minimize(loss)
    #如果是第一次训练，就用sess.run(tf.global_variables_initializer())，
    # 也就不要用到 module_file = tf.train.latest_checkpoint() 和saver.store(sess, module_file)了。
    saver=tf.train.Saver(tf.global_variables(),max_to_keep=15)
    # module_file = tf.train.latest_checkpoint('./')
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # saver.restore(sess, module_file)
        #重复训练2000次
        for i in range(2000):
            for step in range(len(batch_index)-1):
                _,loss_=sess.run([train_op,loss],feed_dict={X:train_x[batch_index[step]:batch_index[step+1]],Y:train_y[batch_index[step]:batch_index[step+1]]})
            print(i, loss_)
            if i % 200==0:
                print("保存模型:",saver.save(sess,'model/stock.model',global_step=i))
with tf.variable_scope('train'):
    train_lstm()
##checkpoint文件会记录保存信息，通过它可以定位最新保存的模型：
## .meta文件保存了当前图结构
## .index文件保存了当前参数名
## .data文件保存了当前参数值
#————————————————预测模型————————————————————
def prediction(time_step=20):
    print("预测模型部分")
    X=tf.placeholder(tf.float32, shape=[None,time_step,input_size])
    #Y=tf.placeholder(tf.float32, shape=[None,time_step,output_size])
    mean,std,test_x,test_y=get_test_data(time_step,5000)
    #第二次定义神经网络变量
    pred,_=lstm(X)
    saver=tf.train.Saver(tf.global_variables())#tf.global_variables的功能均为获取程序中的变量
    # saver = tf.train.import_meta_graph('./model')
    with tf.Session() as sess:
        #参数恢复
        module_file = tf.train.latest_checkpoint('model')
        saver.restore(sess,module_file)
        test_predict=[]
        for step in range(len(test_x)-1):
          prob=sess.run(pred,feed_dict={X:[test_x[step]]})
          predict=prob.reshape((-1))
          test_predict.extend(predict)
        test_y=np.array(test_y)*std[6]+mean[6]
        test_predict=np.array(test_predict)*std[6]+mean[6]
        acc=np.average(np.abs(test_predict-test_y[:len(test_predict)])/test_y[:len(test_predict)])  #偏差
        test_predict2=[]
        # print("偏差值:",end='')
        # print(acc)
        # num=0
        # print(len(test_y))
        # print(len(test_predict))
        # print(test_y)
        # print(test_predict)
        for i in test_predict:
            if i>=0:
                test_predict2.append(1)
            else :
                test_predict2.append(-1)
        plt.figure(figsize=(20,8),dpi=80)
        plt.plot(list(range(len(test_predict2))), test_predict2, color='b')
        plt.plot(list(range(len(test_y))), test_y, color='r')
        plt.show()
        num=0
        for i in range(0,len(test_predict2)):
            if test_predict2[i]!=test_y[i]:
                num+=1
        print("准确率为:%.2f"%(round(num/len(test_y),2)))


#想用混淆矩阵，但是 test_y，test_predict2变量报错，
# cm = confusion_matrix(test_y, test_predict2)     #这不应该是数组么？为啥不行？
# conf_matrix = pd.DataFrame(cm, index=["0", "1"], columns=["0", "1"])
# print(conf_matrix)
#
# dataTemp = pd.DataFrame(columns=columnsNames, index=["0"])
# dataTemp["list"] = str(list)
# dataTemp["Weighted_precision"] = precision_score(test_y, y_pred, average='weighted')
# dataTemp["Weighted_recall"] = recall_score(y_true, y_pred, average='weighted')
# dataTemp["Weighted_f1-score"] = f1_score(y_true, y_pred, average='weighted')

with tf.variable_scope('train',reuse=True):
    prediction(20)


