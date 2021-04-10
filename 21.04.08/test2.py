# encoding: utf-8


import random


# 功能：生成初始化种群
# 参数：personNum为种群数量，length为种群每个个体编码的位数
def initialPopulation(personNum=50, length=17):
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


print(initialPopulation(50, 17))
print(type(initialPopulation(50, 17)[0]))
# print(decode(initialPopulation(personNum,length)[0]))
