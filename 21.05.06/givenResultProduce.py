import pandas as pd
import itertools
import numpy as np

A = []
B = []
C = []
D = []
E = []


def getGivenResult():
    for i in range(1, 151):
        A.append(i)
    for i in range(151, 431):
        B.append(i)
    for i in range(431, 556):
        C.append(i)
    for i in range(556, 801):
        D.append(i)
    for i in range(801, 1001):
        E.append(i)

    return A, B, C, D, E


def getGroupResult(listAddr):
    g1, g2, g3, g4, g5 = getGivenResult()
    group1 = pd.DataFrame(columns=g1, index=['G'])
    group1[g1] = listAddr[0]
    group2 = pd.DataFrame(columns=g2, index=['G'])
    group2[g2] = listAddr[1]
    group3 = pd.DataFrame(columns=g3, index=['G'])
    group3[g3] = listAddr[2]
    group4 = pd.DataFrame(columns=g4, index=['G'])
    group4[g4] = listAddr[3]
    group5 = pd.DataFrame(columns=g5, index=['G'])
    group5[g5] = listAddr[4]
    data = group1.join(group2).join(group3).join(group4).join(group5)
    return data


def fullArrangement():
    array = [0, 1, 2, 3, 4]
    listA = []
    pailie = list(itertools.permutations(array))  # 要list一下，不然它只是一个对象
    for x in pailie:
        for y in x:
            listA.append(y)
        print(listA)
        listA.clear()
        print()


fullArrangement()

data = getGroupResult([0, 1, 2, 3, 4])

print(data)
print(np.array(data))
