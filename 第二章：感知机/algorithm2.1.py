# @Time : 2022-03-03 14:29
# @Author : Phalange
# @File : algorithm2.1.py
# @Software: PyCharm
# C'est la vie,enjoy it! :D
import numpy as np
from sklearn.datasets import load_iris
import sklearn
from sklearn.linear_model import Perceptron
import pandas as pd
import matplotlib.pyplot as plt

class Model1:
    def __init__(self,train_x,train_y):
        assert len(train_x) == len(train_y)
        self.w = np.ones(len(train_x[0]),dtype=np.float32)
        self.b = 0
        self.train_x = train_x
        self.train_y = train_y
        self.lr = 0.1
    def fit(self):
        is_wrong = False
        while not is_wrong:
            wrong_count = 0
            for i,x in enumerate(self.train_x):
                if self.train_y[i]*self.sign(x) <=0:
                    self.w +=self.lr* np.dot(self.train_y[i],x)
                    self.b +=self.lr * self.train_y[i]
                    wrong_count +=1
            if wrong_count ==0:
                is_wrong = True
        print(self.w,self.b)
        return 'success!'

    def sign(self,X):
        return 1 if np.dot(self.w,X) +self.b >= 0 else -1


if __name__ == "__main__":

    # load data
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['label'] = iris.target
    data = np.array(df.iloc[:100, [0, 1, -1]])
    X, y1 = data[:, :-1], data[:, -1]
    y1 = np.array([1 if i == 1 else -1 for i in y1])

    x1 = np.array([3,3]).transpose()
    x2 = np.array([4,3]).transpose()
    x3 = np.array([1,1]).transpose()
    x = np.array([x1,x2,x3])
    y = np.array([1,1,-1])
    Model = Model1(x,y)
    print(Model.fit())
    print(Model.sign(x1))
    print(Model.sign(x2))
    print(Model.sign(x3))

    # 试试鸢尾花数据集
    x_points = np.linspace(4, 7, 10)
    Model2 = Model1(X,y1)
    Model2.fit()
    y_ = -(Model2.w[0] * x_points + Model2.b) / Model2.w[1]
    plt.plot(x_points, y_)

    plt.plot(data[:50, 0], data[:50, 1], 'bo', color='blue', label='0')
    plt.plot(data[50:100, 0], data[50:100, 1], 'bo', color='orange', label='1')
    plt.xlabel('sepal length')
    plt.ylabel('sepal width')
    plt.legend()
    plt.show()

    """ sklearn实现"""
    clf = Perceptron(fit_intercept=True,max_iter=1000,shuffle=True)
    clf.fit(X,y1)