# @Time : 2022-03-10 17:40
# @Author : Phalange
# @File : 线性回归和线性分类器.py
# @Software: PyCharm
# C'est la vie,enjoy it! :D

import warnings
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.preprocessing import PolynomialFeatures
import pandas as pd


warnings.filterwarnings('ignore')

def sigma(z):
    return 1. / (1 + np.exp(-z))


xx = np.linspace(-10, 10, 1000)
plt.plot(xx, [sigma(x) for x in xx])
plt.xlabel('z')
plt.ylabel('sigmoid(z)')
plt.title('Sigmoid function')
plt.show()

# 读取数据集 芯片数据集
data = pd.read_csv('https://labfile.oss.aliyuncs.com/courses/1283/microchip_tests.txt',
                   header=None, names=('test1', 'test2', 'released'))
# 查看数据集的一些信息
data.info()

X = data.iloc[:, :2].values
y = data.iloc[:, 2].values

# 绘制数据，橙点是有缺陷的点
plt.scatter(X[y == 1, 0], X[y == 1, 1], c='blue', label='Released')
plt.scatter(X[y == 0, 0], X[y == 0, 1], c='orange', label='Faulty')
plt.xlabel("Test 1")
plt.ylabel("Test 2")
plt.title('2 tests of microchips. Logit with C=1')
plt.legend()
#plt.show()

# 定义个函数来显示分类器的分界线
def plot_boundary(clf, X, y, grid_step=.01, poly_featurizer=None):
    x_min, x_max = X[:, 0].min() - .1, X[:, 0].max() + .1
    y_min, y_max = X[:, 1].min() - .1, X[:, 1].max() + .1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, grid_step),
                         np.arange(y_min, y_max, grid_step))

    # 在 [x_min, m_max]x[y_min, y_max] 的每一点都用它自己的颜色来对应
    Z = clf.predict(poly_featurizer.transform(np.c_[xx.ravel(), yy.ravel()]))
    Z = Z.reshape(xx.shape)
    plt.contour(xx, yy, Z, cmap=plt.cm.Paired)

# 逻辑回归的实现
poly = PolynomialFeatures(degree=7)
X_poly = poly.fit_transform(X)

print(X_poly.shape)

# 训练逻辑回归模型
# C的值为正则化系数，可以提高C，例如C=1，就会得到不一样的结果
# 可以认为，C对应了模型的复杂度，C越大，正则化越弱
C = 1e-2
logit = LogisticRegression(C=C,random_state=17)
logit.fit(X_poly,y)

plot_boundary(logit,X,y,grid_step=.01,poly_featurizer=poly)
plt.scatter(X[y == 1, 0], X[y == 1, 1], c='blue', label='Released')
plt.scatter(X[y == 0, 0], X[y == 0, 1], c='orange', label='Faulty')
plt.xlabel("Test 1")
plt.ylabel("Test 2")
plt.title('2 tests of microchips. Logit with C=%s' % C)
plt.legend()

print("Accuracy on training set:",
      round(logit.score(X_poly, y), 3))

# LogisticRegressionCV() 方法进行网格搜索参数后再交叉验证，这个方法是专门为逻辑回归设计的
# 该单元格执行时间较长，请耐心等待
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=17)
# 下方结尾的切片为了在线上环境搜索更快，线下练习时可以删除
c_values = np.logspace(-2, 3, 500)[50:450:50]
logit_searcher = LogisticRegressionCV(
    Cs=c_values,cv=skf,verbose=1,n_jobs=1)
logit_searcher.fit(X_poly,y)

