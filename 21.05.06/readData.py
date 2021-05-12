import pandas as pd

users = []
products = []

fileName = "data/score.txt"
replaceNaNAsZero = True


def readDataAsDataFrame():
    file = open(fileName, "r")
    for line in file.readlines():
        line = line.strip()
        user = int(line.split(',')[0])
        product = int(line.split(',')[1])
        if not user in users:
            users.append(user)
        if not product in products:
            products.append(product)
    # 关闭文件
    file.close()
    users.sort()
    products.sort()
    data = pd.DataFrame(index=products, columns=users)
    file = open(fileName, "r")
    for line in file.readlines():
        line = line.strip()
        user = int(line.split(',')[0])
        product = int(line.split(',')[1])
        sorce = float(line.split(',')[2])
        data[user][product] = sorce
    # 关闭文件
    file.close()
    data = data.fillna(data.mean())
    return data


# data = readDataAsDataFrame()
# print(data)