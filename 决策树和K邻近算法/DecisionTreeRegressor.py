# @Time : 2022-03-09 20:45
# @Author : Phalange
# @File : DecisionTreeRegressor.py
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
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import export_graphviz
sns.set()
warnings.filterwarnings('ignore')

n_train = 150
n_test = 1000
noise = 0.1
def f(x):
    x = x.ravel() # 散开数组
    return np.exp(-x ** 2) + 1.5 * np.exp(-(x - 2) ** 2)

def generate(n_samples,noise):
    X = np.random.rand(n_samples) * 10 -5
    X = np.sort(X).ravel()
    y = np.exp(-X ** 2) + 1.5 * np.exp(-(X - 2) ** 2) + \
        np.random.normal(0.0,noise,n_samples)

    X = X.reshape((n_samples,1))
    return X,y

X_train,y_train = generate(n_samples=n_train,noise=noise)
X_test,y_test = generate(n_samples=n_test,noise=noise)


reg_tree = DecisionTreeRegressor(max_depth=5,random_state=17)

reg_tree.fit(X_train,y_train)
reg_tree_pred = reg_tree.predict(X_test)

