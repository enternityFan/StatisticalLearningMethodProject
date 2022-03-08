# @Time : 2022-03-03 16:09
# @Author : Phalange
# @File : algorithm3.1.py
# @Software: PyCharm
# C'est la vie,enjoy it! :D

import numpy as np
import math
from itertools import combinations
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from collections import Counter
from sklearn.neighbors import KNeighborsClassifier

def L(x,y,p=2):
    assert len(x) == len(y) and len(x) >1
    sum = 0
    for i in range(len(x)):
        sum +=math.pow(abs(x[i] - y[i]),p)
    return math.pow(sum,1/p)

class KNN:
    def __init__(self,X_train,y_train,n_neighbors=3,p=2):
        """
                parameter: n_neighbors 临近点个数
                parameter: p 距离度量
        """
        self.n = n_neighbors
        self.p = p
        self.X_train = X_train
        self.y_train = y_train

    def predict(self,X):
        # 取出n个点
        knn_list = []
        for i in range(self.n):
            dist = np.linalg.norm(X - self.X_train[i],ord = self.p)
            knn_list.append((dist,self.y_train[i]))

        # 找与x最相邻的k个点
        for i in range(self.n,len(self.X_train)):
            max_index = knn_list.index(max(knn_list,key = lambda x: x[0]))
            dist = np.linalg.norm(X - self.X_train[i],ord=self.p)
            if knn_list[max_index][0] > dist:
                knn_list[max_index] = (dist,self.y_train[i])

        # 统计
        knn = [k[-1] for k in knn_list]
        count_pairs = Counter(knn)
        max_count = sorted(count_pairs.items(),key=lambda x: x[1])[-1][0]
        return max_count

    def score(self,X_test,y_test):
        right_count = 0
        n = 10
        for X,y in zip(X_test,y_test):
            label = self.predict(X)
            if label == y:
                right_count +=1
        return right_count / len(X_test)


if __name__ == "__main__":

    x1 = [1,1]
    x2 = [5,1]
    x3 = [4,4]

    for i in range(1, 5):
        r = {'1-{}'.format(c): L(x1, c, p=i) for c in [x2, x3]}
        print(min(zip(r.values(), r.keys())))

    # 找出n个距离最近的点的分类情况，少数服从多数
    iris = load_iris()
    df = pd.DataFrame(iris.data,columns=iris.feature_names)
    df['label'] = iris.target
    df.columns = ['sepal length', 'sepal width', 'petal length', 'petal width', 'label']
    #print(df)
    data = np.array(df.iloc[:100, [0, 1, -1]])
    X, y = data[:, :-1], data[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    clf = KNN(X_train, y_train)
    print(clf.score(X_test,y_test))
    test_point = [6.0, 3.0]
    print('Test Point: {}'.format(clf.predict(test_point)))
    plt.scatter(df[:50]['sepal length'], df[:50]['sepal width'], label='0')
    plt.scatter(df[50:100]['sepal length'], df[50:100]['sepal width'], label='1')
    plt.plot(test_point[0], test_point[1], 'bo', label='test_point')
    plt.xlabel('sepal length')
    plt.ylabel('sepal width')
    plt.legend()
    plt.show()

    """scikit-learn"""
    clf_sk = KNeighborsClassifier()
    clf_sk.fit(X_train,y_train)
    print(clf_sk.score(X_train,y_train))
