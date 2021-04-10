# encoding: utf-8
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
import numpy as np


def getMAE(data, mid):
    """
    计算适应度函数?

    :param data:
    :param mid:
    :return:返回计算后的适应度值  平均绝对误差
    """
    x_train = data.iloc[1:mid, 0:5].values  # 训练集?
    x_test = data.iloc[mid:-1, 0:5].values  # 测试集?

    # 总辐射
    y1_train = data.iloc[1:mid, 6].values * 60
    y1_test = data.iloc[mid:-1, 6].values * 60

    # 直辐射
    y2_train = data.iloc[1:mid, 8].values * 60
    y2_test = data.iloc[mid:-1, 8].values * 60

    # 反辐射
    y3_train = data.iloc[1:mid, 10].values * 60
    y3_test = data.iloc[mid:-1, 10].values * 60

    # 散辐射
    y4_train = data.iloc[1:mid, 12].values * 60
    y4_test = data.iloc[mid:-1, 12].values * 60

    scaler_x = StandardScaler()  # 创建归一化的类
    scaler_x.fit(x_train)  # 拟合数据
    x_train = scaler_x.transform(x_train)  # 数据归一化
    x_test = scaler_x.transform(x_test)  # 数据归一化

    scaler_y1 = StandardScaler()  # 创建归一化的类
    scaler_y1.fit(y1_train.reshape(-1, 1))  # 拟合数据
    y1_train = scaler_y1.transform(y1_train.reshape(-1, 1))  # 数据归一化
    y1_test = scaler_y1.transform(y1_test.reshape(-1, 1))  # 数据归一化

    scaler_y2 = StandardScaler()  # 创建归一化的类
    scaler_y2.fit(y2_train.reshape(-1, 1))  # 拟合数据
    y2_train = scaler_y2.transform(y2_train.reshape(-1, 1))  # 数据归一化
    y2_test = scaler_y2.transform(y2_test.reshape(-1, 1))  # 数据归一化

    scaler_y3 = StandardScaler()  # 创建归一化的类
    scaler_y3.fit(y3_train.reshape(-1, 1))  # 拟合数据
    y3_train = scaler_y3.transform(y3_train.reshape(-1, 1))  # 数据归一化
    y3_test = scaler_y3.transform(y3_test.reshape(-1, 1))  # 数据归一化

    scaler_y4 = StandardScaler()  # 创建归一化的类
    scaler_y4.fit(y4_train.reshape(-1, 1))  # 拟合数据
    y4_train = scaler_y4.transform(y4_train.reshape(-1, 1))  # 数据归一化
    y4_test = scaler_y4.transform(y4_test.reshape(-1, 1))  # 数据归一化

    svr = GridSearchCV(SVR(kernel='rbf', gamma=0.1),  # 对那个算法寻优
                       param_grid={"C": [1e0, 1e1, 1e2, 1e3],
                                   "gamma": np.logspace(-2, 2, 5)})

    svr.fit(x_train, y1_train)  # 训练模型
    linear_svr_y_predict_gv = svr.predict(x_test)  # 预测测试集

    y1_test_org = scaler_y1.inverse_transform(y1_test)  # 将标准化后的数据转换为原始数据
    linear_svr_y_predict_org1_gv = scaler_y1.inverse_transform(linear_svr_y_predict_gv)  # 将预测后的结果转换为(原始数据)
    # print("MAE: {}".format(mean_absolute_error(y1_test_org,linear_svr_y_predict_org1_gv)))
    return mean_absolute_error(y1_test_org, linear_svr_y_predict_org1_gv)  # 计算 平均绝对误差 并返回
