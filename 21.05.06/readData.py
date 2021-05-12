import pandas as pd

users = []
products = []

file = open("data/score.txt", "r")

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



data = pd.DataFrame(index=users,columns=products)
file = open("data/score.txt", "r")

for line in file.readlines():
    line = line.strip()
    user = int(line.split(',')[0])
    product = int(line.split(',')[1])
    sorce = float(line.split(',')[2])
    data[product][user] = sorce
# 关闭文件
file.close()

print(data)