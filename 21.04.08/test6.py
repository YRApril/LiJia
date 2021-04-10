# encoding: utf-8
from aJPloy import evaluate

import random

mutationProbability = 0.6  # 编译概率
# length = 29
length = 17


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


print('00000000010000111')
print(getVari('00000000010000111'))
