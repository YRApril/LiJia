import numpy as np
import random
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
import readData
import givenResultProduce
from matplotlib import pyplot as plt

# 用k-means++进行分类
# X = np.array([[0, 2], [0, 0], [1, 0], [5, 0], [5, 2]])
X = readData.get2DimensionValue(readData.readDataAsDataFrame())

# 模型创建
kmeans_model = KMeans(n_clusters=5, init='k-means++', random_state=11)
# 进行聚类处理
y_pred = kmeans_model.fit_predict(X)
# print("y_pred", y_pred)

# 评估分类结果
# result_evalue = pd.DataFrame(columns='Weighted_precision,Weighted_recall,Weighted_f1-score,Macro_precision,Macro_recall,Macro_f1-score,Micro_precision,Micro_recall,Micro_f1-score', index=['G'])
# print(result_evalue)
lists = givenResultProduce.fullArrangement()

columnsNames = ["list", "Weighted_precision", "Weighted_recall", "Weighted_f1-score", "Macro_precision", "Macro_recall",
                "Macro_f1-score", "Micro_precision", "Micro_recall", "Micro_f1-score"]

resultDimen = pd.DataFrame(columns=columnsNames)

for list in lists:
    print()
    print(list)
    y_true = np.array(givenResultProduce.getGroupResult(list))
    # print("y_true1111:\n", y_true)
    y_true = y_true.flatten()
    # print("y_true2222:\n", y_true)

    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)
    cm = confusion_matrix(y_true, y_pred)
    conf_matrix = pd.DataFrame(cm, index=["0", "1", "2", "3", "4"], columns=["0", "1", "2", "3", "4"])
    print(conf_matrix)
    print()

    dataTemp = pd.DataFrame(columns=columnsNames, index=["0"])
    dataTemp["list"] = str(list)
    dataTemp["Weighted_precision"] = precision_score(y_true, y_pred, average='weighted')
    dataTemp["Weighted_recall"] = recall_score(y_true, y_pred, average='weighted')
    dataTemp["Weighted_f1-score"] = f1_score(y_true, y_pred, average='weighted')
    dataTemp["Macro_precision"] = precision_score(y_true, y_pred, average='macro')
    dataTemp["Macro_recall"] = recall_score(y_true, y_pred, average='macro')
    dataTemp["Macro_f1-score"] = f1_score(y_true, y_pred, average='macro')
    dataTemp["Micro_precision"] = precision_score(y_true, y_pred, average='micro')
    dataTemp["Micro_recall"] = recall_score(y_true, y_pred, average='micro')
    dataTemp["Micro_f1-score"] = f1_score(y_true, y_pred, average='micro')

    resultDimen = resultDimen.append(dataTemp, ignore_index=True)

    # print('------Weighted-----')
    # print('Weighted_precision',precision_score(y_true,y_pred,average='weighted'))
    # print('Weighted_recall', recall_score(y_true, y_pred, average='weighted'))
    # print('Weighted_f1-score', f1_score(y_true, y_pred, average='weighted'))
    #
    # print('------Macro-----')
    # print('Macro_precision', precision_score(y_true, y_pred, average='macro'))
    # print('Macro_recall', recall_score(y_true, y_pred, average='macro'))
    # print('Macro_f1-score', f1_score(y_true, y_pred, average='macro'))
    #
    # print('------Micro-----')
    # print('Micro_precision', precision_score(y_true, y_pred, average='micro'))
    # print('Micro_recall', recall_score(y_true, y_pred, average='micro'))
    # print('Micro_f1-score', f1_score(y_true, y_pred, average='micro'))

    plt.matshow(conf_matrix, cmap=plt.gray())
    plt.title(str(list))
    plt.savefig("out/" + str(list) + ".png")
    plt.show()

print(resultDimen)
resultDimen.to_csv("out/result.csv")
