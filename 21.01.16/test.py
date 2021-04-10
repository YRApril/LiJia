import matplotlib.pyplot as plt
import numpy as np
import random
import pandas as pd

# plt.plot([1,2,3,4])
# plt.ylabel('some numbers')
# plt.show()

# plt.plot([1,2,3,4],[1,4,9,16],'ro')
# plt.axis([0,6,0,20])
# plt.show()

# 0到5之间每隔0.2取一个数
# t=np.arange(0.,5.,0.2)

# 红色的破折号，蓝色的方块，绿色的三角星
# plt.plot(t,t,'r--',t,t**2,'bs',t,t**3,'g^')
# plt.show()

a = np.array([[1, 1],
              [1, 2],
              [2, 1],
              [6, 4],
              [6, 3],
              [5, 4]])

clalist = np.array([[0., 5.],
                    [1., 4.47213595],
                    [1., 4.24264069],
                    [5.83095189, 1.],
                    [5.38516481, 1.41421356],
                    [5., 0.]])

print("clalist:")
print(clalist)
print()

# print(np.tile(a,(2,1)))
# minDistIndices=[[0 ,0,0 ,1, 1, 1]]


minDistIndices = np.argmin(clalist, axis=1)  # 水平方向 最小值的下标

print("minDistIndices:")
print(minDistIndices)
print()

print("type(minDistIndices):" + str(type(minDistIndices)) + "\n")

newCentroids = pd.DataFrame(a)
print("newCentroids:")
print(newCentroids)
print('=================')

print("pd.DataFrame(a):")
print(pd.DataFrame(a))
print()

print("minDistIndices:")
print(minDistIndices)
print()

newCentroids = pd.DataFrame(a).groupby(minDistIndices).mean()
print("pd.DataFrame(a).groupby(minDistIndices).mean()")
print(newCentroids)
print()

print("pd.DataFrame(a).groupby(minDistIndices).sum()")
print(pd.DataFrame(a).groupby(minDistIndices).sum())
print()

dd = pd.DataFrame(a)
print("pd.DataFrame(a):")
print(dd)
print()

dd0 = dd.groupby(0)
print("dd0.mean():")
print(dd0.mean())
print()

tryddd = np.array([0, 0, 0, 1, 1, 1])

print(tryddd)
print(type(tryddd))
print()

print(dd.groupby(tryddd).mean())
print()

print(np.array(pd.DataFrame(a).groupby(minDistIndices)))
print()

print(pd.DataFrame(pd.DataFrame(a).groupby(minDistIndices)))
print()

