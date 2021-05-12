import readData
from sklearn.manifold import TSNE

data = readData.readDataAsDataFrame()

data = data.fillna(0)

print(data)

tsne = TSNE()
x_temp_trans = tsne.fit_transform(data)
print(x_temp_trans)