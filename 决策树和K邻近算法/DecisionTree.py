# @Time : 2022-03-09 20:33
# @Author : Phalange
# @File : DecisionTree.py
# @Software: PyCharm
# C'est la vie,enjoy it! :D

import warnings
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from ipywidgets import Image
from io import StringIO
import pydotplus
from sklearn.tree import export_graphviz
sns.set()
warnings.filterwarnings('ignore')

def get_grid(data):
    x_min,x_max = data[:,0].min() - 1,data[:,0].max() + 1
    y_min,y_max = data[:,1].min() - 1,data[:,1].max() + 1
    return np.meshgrid(np.arange(x_min,x_max,0.01),np.arange(y_min,y_max,0.01))

# 生成两个不同类别的数据

# 第一类数据
np.random.seed(17)
train_data = np.random.normal(size=(100,2))
train_labels = np.zeros(100)

# 第二类
train_data = np.r_[train_data,np.random.normal(size=(100,2),loc=2)] # 二维数组，则纵向堆叠（列不变，行增加）。
train_labels = np.r_[train_labels,np.ones(100)]

# 查看数据
plt.figure(figsize=(10, 8))
plt.scatter(train_data[:, 0], train_data[:, 1], c=train_labels, s=100,
            cmap='autumn', edgecolors='black', linewidth=1.5)
plt.plot(range(-2, 5), range(4, -3, -1))
#plt.show()

clf_tree = DecisionTreeClassifier(criterion='entropy',max_depth=3,random_state=17)
clf_tree.fit(train_data,train_labels)
# 可视化
xx, yy = get_grid(train_data)
predicted = clf_tree.predict(np.c_[xx.ravel(),
                                   yy.ravel()]).reshape(xx.shape)
plt.pcolormesh(xx, yy, predicted, cmap='autumn')
plt.scatter(train_data[:, 0], train_data[:, 1], c=train_labels, s=100,
            cmap='autumn', edgecolors='black', linewidth=1.5)
#plt.show()
# 显示决策树
dot_data = StringIO()
export_graphviz(clf_tree, feature_names=['x1', 'x2'],
                out_file=dot_data, filled=True)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
Image(value=graph.create_png())

