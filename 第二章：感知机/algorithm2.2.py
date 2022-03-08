# @Time : 2022-03-03 15:10
# @Author : Phalange
# @File : algorithm2.2.py
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
        self.w = np.ones((len(train_x),1),dtype=np.float32)
        self.train_x = train_x
        self.train_y = train_y
        self.Gram = np.dot(train_x,np.transpose(train_x))
        self.lr = 0.1
        self.b = 0


    def fit(self):
        is_wrong = False
        while is_wrong == False:
            count = 0
            for i in range(len(self.train_x)):
               if  self.train_y[i]* self.sign(i) <=0:
                   self.w[i] +=self.lr
                   self.b +=self.lr*self.train_y[i]
                   count +=1

            if count == 0:
                is_wrong = True
        print(self.w,self.b)
        return




    def sign(self,i):
        y =np.dot(self.train_y.reshape(1,-1)*np.transpose(self.w),self.Gram[:,i]) + self.b
        return -1 if y<0 else 1

if __name__ == "__main__":


    # 我的实现，并且算书上的例题
    x1 = np.array([3,3]).transpose()
    x2 = np.array([4,3]).transpose()
    x3 = np.array([1,1]).transpose()
    x = np.array([x1,x2,x3])
    y = np.array([1,1,-1])
    Model = Model1(x,y)
    print(Model.fit())
    print(Model.sign(2))
    print(Model.sign(1))
    print(Model.sign(0))
