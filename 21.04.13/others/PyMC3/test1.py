import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

np.random.seed(123)

alpha=1
sigma=1
beta =[1, 2.5]

N=100

X1=np.random.randn(N)
X2=np.random.randn(N)

Y=alpha + beta[0]*X1 + beta[1]*X2 + np.random.randn(N)*sigma

%matplotlib inline
fig1,ax1 = plt.subplots(1, 2, figsize=(10,4));
ax1[0].scatter(X1, Y);ax1[0].set_xlabel('X1');ax1[0].set_ylabel('Y');
ax1[1].scatter(X2, Y);ax1[1].set_xlabel('X2');ax1[1].set_ylabel('Y');

fig2 = plt.figure(2);
ax2 = Axes3D(fig2);
ax2.scatter(X1,X2,Y);
ax2.set_xlabel('X1');
ax2.set_ylabel('X2');
ax2.set_zlabel('Y');