import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# 定义读取文件的函数
def read_data(file_path):
    # column_names所有列表的名称
    column_names = ['label', 'x_value', 'y_value']
    data = pd.read_csv(file_path, header=None, names=column_names)
    # 按列分离数据
    # label=data[['label']]
    # print(label)
    # x=data[['x_value']]
    # print(x)
    # y=data[['y_value']]
    # print(y)
    return data


# 画出带有标签的彩图
def scatter_with_color(x, y, labels, figure_no):
    plt.scatter(x, y, c='b')
    plt.figure(figure_no)
    plt.xlabel('x values')
    plt.ylabel('y values')
    plt.title('Scatter with color')


# 画线性图
# def simple_line_plot(x,y,figure_no):
#  plt.figure(figure_no)
#  plt.plot(x,y)
#  plt.xlabel('x values')
#  plt.ylabel('y values')
#  plt.title('simple_line_plot')

# 调用函数读取数据
dataset = read_data('data/testSet(1).csv')
dataset.drop([0], inplace=True)
# print(dataset)
figure_no = 1
x = dataset['x_value']
y = dataset['y_value']
label = dataset['label']
# scatter_with_color(x, y, label, figure_no)
scatter_with_color(list(map(int, x.tolist())), list(map(int, y.tolist())), label, figure_no)
# figure_no+=1
# simple_line_plot(x,y,figure_no)
plt.show()
