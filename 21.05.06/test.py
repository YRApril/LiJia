import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn import datasets
import pandas as pd

inputfile = 'data/data111.csv'
data = pd.read_csv(inputfile)
dataset = data.values
print(dataset)