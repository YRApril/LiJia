# encoding: utf-8
import time

from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from testData import getData
import numpy as np
import random

personNum = 50  # 种群大小
length = 17  # 个体长度
mutationProbability = 0.6  # 变异概率


def decode(onePerson):
    """
    将一个17位的二进制编码转换为C,gamma的十进制解

    前十位表示C【0，10000】

    后七位表示gamma（0，100】

    :param onePerson:要转换的二进制数，17位，前10位为C，后7位为gamma
    :return:（C，gamma）
    """
    """首先C和gamma都应该是个连续值，但是计算机表示的数据都是离散的。那么，针对这个问题，我是这样做的：
首先，对于C【0，1w】的取值，我们可以把它分为1000份，每一份代表10，也就是说表示范围变为：
0,10, 20, 30 一直到1w。
我们知道，2的10次方是1024，所以我们可以用10位二进制来表示这1000份，当大于1000时，仍表示为1000。
接下来表示gamma（0，100】，我们知道2的7次方是128，所以我们可以用7位二进制来表示，其中当二进制表的结果小于等于1时，则为1。当二进制表示结果大于等于100时，仍表示100。
那么个体编码就是17位二进制了。"""
    x = onePerson[0:10]  # 前十位表示惩罚因子C
    y = onePerson[10:17]  # 后七位表示gamma
    x = int(x, 2)  # 二进制转换为十进制int型
    y = int(y, 2)  # 二进制转换为十进制int型

    "边界条件"
    if x >= 10000:
        x = 10000
    if y <= 1:
        y = 1
    elif y >= 10000:
        y = 100

    return x, y


def initialPopulation(personNum, length):
    """
    生成初始化种群

    :param personNum: 种群数量
    :param length: 种群每个个体编码的位数
    :return: 返回生成的初始化种群列表 list类型
    """
    totalPopulation = []  # 完整种群空列表
    while len(totalPopulation) != personNum:  # 种群中个体数量不足时，随机生成个体并添加到种群
        person = []  # 要添加到种群中的空白个体，列表
        for i in range(length):  # 根据个体位数循环生成随机数
            temp = random.uniform(-1, 1)  # 生成-1<=X<=1的数字
            if temp < 0:
                person.append(0)
            else:
                person.append(1)
        theStr = ''  # 要添加到种群中的个体，根据列表转换的字符串
        for item in person:
            theStr += str(item)
        # print(theStr)
        "去重，重复个体不放入种群"
        if theStr not in totalPopulation:
            totalPopulation.append(theStr)
    return totalPopulation


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


def evaluate(data, mid, onePerson):
    """
    计算x,y对应的函数值

    :param data:
    :param mid:
    :param onePerson: 单个个体编码
    :return:
    """
    x, y = decode(onePerson)  # 编码转换
    startTime = time.time()
    result = getMAE(data, mid)
    print('训练用时：' + str(round((time.time() - startTime) / 60, 2)) + 'min\n')
    return 1 / result


def getParents(evalList):
    """
    从种群中随机获取一位父母进行交叉？？？？

    :param evalList:
    :return: 返回一对双亲在population的index
    """
    temp = random.uniform(0, 1)
    portionList = []
    theSum = 0
    totalEval = sum(evalList)
    for eval in evalList:
        theSum += eval / totalEval
        portionList.append(theSum)
    location = 0
    while (temp > portionList[location]):
        location += 1
    return location


def getCross(father, mother):
    """
    双亲交叉生成子代

    :param father: 父亲的编码
    :param mother: 母亲的编码
    :return: 根据父母生成的子代的编码
    """
    theVisit = []
    crossLocation = random.randint(0, length - 1)  # 随机选择交叉的位置
    theVisit.append(crossLocation)
    # print(crossLocation)
    child = ''
    child += father[0:crossLocation]  # 交叉点之前由父亲提供
    child += mother[crossLocation:length]  # 交叉点之后由母亲提供

    return child


def getVari(person):
    """
    进行变异

    :param person: 单个个体编码
    :return: 变异后的编码
    """
    # print(person)
    temp = random.uniform(0, 1)
    "生成0 1 之间的随机数，如果小于设定的变异概率，则进行变异，否则直接返回原始个体编码"
    if temp < mutationProbability:
        location = random.randint(0, length - 1)  # 在个体位数内随机产生一个位置,该位置取反进行变异
        tempStr = person[0:location]
        tempStr += str(1 - int(person[location]))
        tempStr += person[location + 1:]
        return tempStr if evaluate(tempStr) > evaluate(person) else person
        # return tempStr if True else person
    return person


if __name__ == '__main__':
    theScore = []  # 最佳个体选中次数
    bestPerson = []  # 最佳个体列表
    theBestEval = 0
    iteration = 20  # 最大代数
    data = getData()
    mid = int(data.shape[0] * 0.8)

    for i in range(20):  # 设置跑多少轮，用来查看收敛性的
        population = initialPopulation(personNum, length)  # 生成初始化种群
        flag = 0  # 代数标记位
        timeStart = time.time()  # 开始时间
        while flag != iteration:  # 没有到最后一代时
            print("第", flag + 1, "代")
            evalList = []  # 评估列表
            tempPopulation = []

            "计算种群中每个个体的效用值,并放入评估列表"
            for person in population:
                evalList.append(evaluate(data, mid, person))

            maxEval = max(evalList)  # 效用值最大值
            theIndex = evalList.index(maxEval)  # 效用值最大值索引
            print('maxEval=', maxEval)

            tempPopulation.append(population[theIndex])  # 每次迭代时先将上一代最大的个体放到下一代种群中

            print("开始交叉")
            for i in range(personNum):
                parentsFaIndex = getParents(evalList)  # 获得用于交叉的父母位置
                parentsFa = population[parentsFaIndex]  # 获得用于交叉的父母

                parentsMaIndex = getParents(evalList)  # 获得用于交叉的父母位置
                parentsMa = population[parentsMaIndex]  # 获得用于交叉的父母

                child = getCross(parentsFa, parentsMa)  # 通过交叉产生子代

                child = getVari(child)  # 子代变异
                tempPopulation.append(child)  # 产生的子代放入下一代

            population = tempPopulation  # 更替代
            flag += 1  # 代数标记位自增

            evalList = []  # 清空评估列表

            "计算新种群中每个个体的效用值,并放入评估列表"
            for person in population:
                evalList.append(evaluate(data, mid, person))

            maxEval = max(evalList)  # 新种群效用值最大值

            "记录效用值最大值"
            if theBestEval < maxEval:
                theBestEval = maxEval

            theIndex = evalList.index(maxEval)  # 效用值最大值索引
            person = population[theIndex]  # 根据效用值最大值索引获取当前代最佳个体

            if person not in bestPerson:  # 当前代最佳个体不在最佳个体列表中时
                bestPerson.append(person)  # 添加
                theScore.append(1)  # 次数对应位写1
            else:
                theScore[bestPerson.index(person)] += 1  # 次数对应位自增
        print('duration=', time.time() - timeStart)

    print(theScore)
    print(bestPerson)

    theBestEvalList = []
    "评估所有代数中的最佳个体"
    for item in bestPerson:
        theBestEvalList.append(evaluate(data, mid, item))
    print(theBestEvalList)
    print(theBestEval)
    print(max(theScore))
