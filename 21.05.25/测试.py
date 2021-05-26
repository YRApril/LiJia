#coding=gbk
'''
Created on 2017��2��20��
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
#���峣��
rnn_unit=10       #hidden layer units
input_size=6     #������Ԫ����
output_size=1    #�����Ԫ����
lr=0.0006         #ѧϰ��
#�������������������������������������������ݡ�������������������������������������������
#df=pd.read_csv("E:\\pythonProject-tensorflow\\��Ʊ����.csv")
df=pd.read_csv("data/��Ʊ����.csv")
# df=pd.read_csv("F:/machineLearning/DateSet/DataAnalysisTask/stock/��Ʊ����.csv")
data=df.iloc[:,1:8].values #ȡ��2-8��,data����һ��6529*7��һ������
#df.shape=(6529,16)
#��ȡѵ����
def get_train_data(batch_size=60,time_step=20,train_begin=0,train_end=5000):
    batch_index=[]
    data_train=data[train_begin:train_end]
    #��׼��Ҳ������һ����һ���ǽ�����ӳ�䵽ָ���ķ�Χ�����ڳ�ȥ��ͬά�����ݵ������Լ����ٵ�λ
    normalized_train_data=(data_train-np.mean(data_train,axis=0))/np.std(data_train,axis=0)  #��׼��
    train_x,train_y=[],[]   #ѵ����
    for i in range(len(normalized_train_data)-time_step):
       #banch_sizeΪÿ����ѵ��������
       if i % batch_size==0:
           batch_index.append(i)
       x=normalized_train_data[i:i+time_step,:6]#ǰ6��Ϊ����ά������
       y=normalized_train_data[i:i+time_step,6,np.newaxis]
       train_x.append(x.tolist())
       train_y.append(y.tolist())
    batch_index.append((len(normalized_train_data)-time_step))
    return batch_index,train_x,train_y
#��ȡ���Լ�
def get_test_data(time_step=20,test_begin=5000):
    data_test=data[test_begin:]
    mean=np.mean(data_test,axis=0)
    std=np.std(data_test,axis=0)
    normalized_test_data=(data_test-mean)/std  #��׼��
    size=(len(normalized_test_data)+time_step-1)//time_step  #��size��sample
    test_x,test_y=[],[]
    for i in range(size-1):
       x=normalized_test_data[i*time_step:(i+1)*time_step,:6]
       y=normalized_test_data[i*time_step:(i+1)*time_step,6]
       test_x.append(x.tolist())
       test_y.extend(y)
    test_x.append((normalized_test_data[(i+1)*time_step:,:6]).tolist())
    test_y.extend((normalized_test_data[(i+1)*time_step:,6]).tolist())
    return mean,std,test_x,test_y
#�������������������������������������������������������������������������������������
#����㡢�����Ȩ�ء�ƫ��
weights={
         'in':tf.Variable(tf.random.normal([input_size,rnn_unit])),
         'out':tf.Variable(tf.random.normal([rnn_unit,1]))
        }
biases={
        'in':tf.Variable(tf.constant(0.1,shape=[rnn_unit,])),
        'out':tf.Variable(tf.constant(0.1,shape=[1,]))
       }
#�������������������������������������������������������������������������������������
def lstm(X):
    batch_size=tf.shape(X)[0]
    time_step=tf.shape(X)[1]
    w_in=weights['in']
    b_in=biases['in']
    input=tf.reshape(X,[-1,input_size])  #��Ҫ��tensorת��2ά���м��㣬�����Ľ����Ϊ���ز������
    input_rnn=tf.matmul(input,w_in)+b_in
    input_rnn=tf.reshape(input_rnn,[-1,time_step,rnn_unit])  #��tensorת��3ά����Ϊlstm cell������
    cell=tf.nn.rnn_cell.BasicLSTMCell(rnn_unit)
    init_state=cell.zero_state(batch_size,dtype=tf.float32)
    output_rnn,final_states=tf.nn.dynamic_rnn(cell, input_rnn,initial_state=init_state, dtype=tf.float32)  #output_rnn�Ǽ�¼lstmÿ������ڵ�Ľ����final_states�����һ��cell�Ľ��
    output=tf.reshape(output_rnn,[-1,rnn_unit]) #��Ϊ����������
    w_out=weights['out']
    b_out=biases['out']
    pred=tf.matmul(output,w_out)+b_out
    return pred,final_states
#������������������������������������ѵ��ģ�͡�����������������������������������
# batch_sizieΪÿ����ѵ����������time_stepΪʱ�䲽,����������������ѵ����������
def train_lstm(batch_size=60,time_step=20,train_begin=0,train_end=5000):
    #tf.placeholder()������Ϊһ��ռλ�����ڶ�����̣��������Ϊ�βΣ���ִ�е�ʱ���ٸ������ֵ��
    #����ָ����ʼֵ����������ʱ��ͨ�� Session.run �ĺ����� feed_dict ����ָ����
    X=tf.placeholder(tf.float32, shape=[None,time_step,input_size])
    Y=tf.placeholder(tf.float32, shape=[None,time_step,output_size])
    batch_index,train_x,train_y=get_train_data(batch_size,time_step,train_begin,train_end)
    #�������������
    pred,_=lstm(X)
    #��ʧ����
    loss=tf.reduce_mean(tf.square(tf.reshape(pred,[-1])-tf.reshape(Y, [-1])))
    train_op=tf.train.AdamOptimizer(lr).minimize(loss)
    #����ǵ�һ��ѵ��������sess.run(tf.global_variables_initializer())��
    # Ҳ�Ͳ�Ҫ�õ� module_file = tf.train.latest_checkpoint() ��saver.store(sess, module_file)�ˡ�
    saver=tf.train.Saver(tf.global_variables(),max_to_keep=15)
    # module_file = tf.train.latest_checkpoint('./')
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # saver.restore(sess, module_file)
        #�ظ�ѵ��2000��
        for i in range(2000):
            for step in range(len(batch_index)-1):
                _,loss_=sess.run([train_op,loss],feed_dict={X:train_x[batch_index[step]:batch_index[step+1]],Y:train_y[batch_index[step]:batch_index[step+1]]})
            print(i, loss_)
            if i % 200==0:
                print("����ģ��:",saver.save(sess,'model/stock.model',global_step=i))
with tf.variable_scope('train'):
    train_lstm()
##checkpoint�ļ����¼������Ϣ��ͨ�������Զ�λ���±����ģ�ͣ�
## .meta�ļ������˵�ǰͼ�ṹ
## .index�ļ������˵�ǰ������
## .data�ļ������˵�ǰ����ֵ
#��������������������������������Ԥ��ģ�͡���������������������������������������
def prediction(time_step=20):
    print("Ԥ��ģ�Ͳ���")
    X=tf.placeholder(tf.float32, shape=[None,time_step,input_size])
    #Y=tf.placeholder(tf.float32, shape=[None,time_step,output_size])
    mean,std,test_x,test_y=get_test_data(time_step,5000)
    #�ڶ��ζ������������
    pred,_=lstm(X)
    saver=tf.train.Saver(tf.global_variables())#tf.global_variables�Ĺ��ܾ�Ϊ��ȡ�����еı���
    # saver = tf.train.import_meta_graph('./model')
    with tf.Session() as sess:
        #�����ָ�
        module_file = tf.train.latest_checkpoint('model')
        saver.restore(sess,module_file)
        test_predict=[]
        for step in range(len(test_x)-1):
          prob=sess.run(pred,feed_dict={X:[test_x[step]]})
          predict=prob.reshape((-1))
          test_predict.extend(predict)
        test_y=np.array(test_y)*std[6]+mean[6]
        test_predict=np.array(test_predict)*std[6]+mean[6]
        acc=np.average(np.abs(test_predict-test_y[:len(test_predict)])/test_y[:len(test_predict)])  #ƫ��
        test_predict2=[]
        # print("ƫ��ֵ:",end='')
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
        print("׼ȷ��Ϊ:%.2f"%(round(num/len(test_y),2)))


#���û������󣬵��� test_y��test_predict2��������
# cm = confusion_matrix(test_y, test_predict2)     #�ⲻӦ��������ô��Ϊɶ���У�
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


