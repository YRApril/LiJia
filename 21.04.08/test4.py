# encoding: utf-8
from test3 import getMAE
from aJPloy import decode


def evaluate(data, mid, onePerson):
    """
    计算x,y对应的函数值
    :param data:
    :param mid:
    :param onePerson: 单个个体编码
    :return:
    """
    x, y = decode(onePerson)
    result = getMAE(data, mid)
    return 1 / result
