# @Time : 2022-03-09 21:01
# @Author : Phalange
# @File : KNN临近法.py
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


def form_linearly_separable_data(n=500,x1_min=0,x1_max=30,
                                 x2_min=0,x2_max=30):
    data,target = [],[]
    for i in range(n):
        x1 = np.random.randint(x1_min,x1_max)
        x2 = np.random.randint(x2_min,x2_max)
        if np.abs(x1 - x2) > 0.5:
            data.append([x1,x2])
            target.append(np.sign(x1 - x2))
    return np.array(data),np.array(target)

def get_grid(data):
    x_min, x_max = data[:, 0].min() - 1, data[:, 0].max() + 1
    y_min, y_max = data[:, 1].min() - 1, data[:, 1].max() + 1
    return np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))



# 在客户离网率预测任务中使用决策树和最近邻方法
df = pd.read_csv(
    'https://labfile.oss.aliyuncs.com/courses/1283/telecom_churn.csv')

df['International plan'] = pd.factorize(df['International plan'])[0]
df['Voice mail plan'] = pd.factorize(df['Voice mail plan'])[0]
df['Churn'] = df['Churn'].astype('int')
states = df['State']
y = df['Churn']
df.drop(['State', 'Churn'], axis=1, inplace=True)


X_train, X_holdout, y_train, y_holdout = train_test_split(df.values, y, test_size=0.3,
                                                          random_state=17)

tree = DecisionTreeClassifier(max_depth=5,random_state=17)
knn = KNeighborsClassifier(n_neighbors=10)

tree.fit(X_train,y_train)
knn.fit(X_train,y_train)

tree_pred = tree.predict(X_holdout)
accuracy_score(y_holdout,tree_pred)

knn_pred = knn.predict(X_holdout)
accuracy_score(y_holdout,tree_pred)
tree_params = {'max_depth': range(5, 7),
               'max_features': range(16, 18)}

# GridSearchCV实现简单的交叉验证
tree_grid = GridSearchCV(tree,tree_params,cv=5,n_jobs=-1,verbose=True)

tree_grid.fit(X_train,y_train)

print(tree_grid.best_params_)

print(tree_grid.best_score_)

accuracy_score(y_holdout,tree_grid.predict(X_holdout))

knn_pipe = Pipeline([('scaler',StandardScaler()),
                     ('knn',KNeighborsClassifier(n_jobs=-1))])

knn_params = {'knn__n_neighbors':range(6,8)}

knn_grid = GridSearchCV(knn_pipe,knn_params,
                        cv=5,n_jobs=-1,verbose=True)

knn_grid.fit(X_train,y_train)

print(knn_grid.best_params_)
print(knn_grid.best_score_)
print(accuracy_score(y_holdout,knn_grid.predict(X_holdout)))

forest = RandomForestClassifier(n_estimators=100,n_jobs=1,
                                random_state=17)
np.mean(cross_val_score(forest,X_train,y_train,cv=5))

forest_params = {'max_depth': range(8, 10),
                 'max_features': range(5, 7)}

forest_grid = GridSearchCV(forest, forest_params,
                           cv=5, n_jobs=-1, verbose=True)

forest_grid.fit(X_train, y_train)
print(forest_grid.best_params_,forest_grid.best_score_)

print(accuracy_score(y_holdout,forest_grid.predict(X_holdout)))


# 决策树的复杂情况
# 生成的数据集可以用x1=x2这么简单的分类就完成
X,y = form_linearly_separable_data()
plt.scatter(X[:,0],X[:,1],c=y,cmap='autumn',edgecolors='black')

# 训练一颗决策树对上面数据进行分类，并绘制分类边界
tree = DecisionTreeClassifier(random_state=17).fit(X,y)

xx,yy = get_grid(X)
predicted = tree.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
plt.pcolormesh(xx, yy, predicted, cmap='autumn')
plt.scatter(X[:, 0], X[:, 1], c=y, s=100,
            cmap='autumn', edgecolors='black', linewidth=1.5)
plt.title('Easy task. Decision tree compexifies everything')
#plt.show()
# 可以发现决策树对这个任务建模的太过复杂，出现了过拟合的情况。

# KNN邻近法
