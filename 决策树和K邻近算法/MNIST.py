# @Time : 2022-03-10 10:32
# @Author : Phalange
# @File : MNIST.py
# @Software: PyCharm
# C'est la vie,enjoy it! :D


import warnings
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, cross_val_score
sns.set()
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_digits

data = load_digits()
X,y = data.data,data.target

# 该数据库中手写数字的图片为 8x8 的矩阵，矩阵中的值表示每个像素的白色亮度。
#print(X[0, :].reshape([8, 8]))

# 绘制一些MNIST手写数字
f, axes = plt.subplots(1, 4, sharey=True, figsize=(16, 6))
for i in range(4):
    axes[i].imshow(X[i, :].reshape([8, 8]), cmap='Greys')

# 分割数据集
X_train, X_holdout, y_train, y_holdout = train_test_split(
    X, y, test_size=0.3, random_state=17)

# 使用随机参数训练决策树和k-NN
tree = DecisionTreeClassifier(max_depth=5,random_state=17)
knn_pipe = Pipeline([('scaler',StandardScaler()),
                     ('knn',KNeighborsClassifier(n_neighbors=10))])
tree.fit(X_train,y_train)
knn_pipe.fit(X_train,y_train)
# 进行预测
tree_pred = tree.predict(X_holdout)
knn_pred = knn_pipe.predict(X_holdout)
print("knn的预测结果：")
print(accuracy_score(y_holdout,knn_pred))
print("tree的预测结果：")
print(accuracy_score(y_holdout,tree_pred))

# 使用交叉验证调优决策树模型
tree_params = {'max_depth':[10,20,30],
               'max_features':[30,50,64]}

tree_grid = GridSearchCV(tree,tree_params,cv=5,n_jobs=1,verbose=True)
tree_grid.fit(X_train,y_train)
print("调优后的决策树的最佳的结果:")
print(tree_grid.best_score_)

# 使用交叉验证调优k-NN
print(np.mean(cross_val_score(KNeighborsClassifier(n_neighbors=1),X_train,y_train,cv=5)))

# 训练随机森林模型，在大多数数据集上，它的效果比k-NN要好
print(np.mean(cross_val_score(RandomForestClassifier(
    random_state=17), X_train, y_train, cv=5)))
# 注意，这个随机森林的结果没有调优

