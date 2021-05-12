import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn import datasets
import pandas as pd

np.set_printoptions(suppress=True)

inputfile = 'data/data111.csv'
data = pd.read_csv(inputfile)

print(data)
dataset = data.values

print(dataset)
print(type(dataset))