# encoding: utf-8


import random

length = 17


def getParents(evalList):
    """
    从种群中随机获取一对父母进行交叉？？？？

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
