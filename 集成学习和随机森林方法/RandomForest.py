# @Time : 2022-03-11 10:44
# @Author : Phalange
# @File : RandomForest.py
# @Software: PyCharm
# C'est la vie,enjoy it! :D
import pandas as pd
from sklearn.model_selection import cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.metrics import accuracy_score

from sklearn.model_selection import train_test_split
from sklearn.datasets import make_circles
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, BaggingRegressor
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import seaborn as sns
from matplotlib import pyplot as plt

# Disable warnings in Anaconda
import warnings
import numpy as np
warnings.filterwarnings('ignore')

# 对于随机森林的算法，分类问题使用多数投票算法，回归问题使用均值,分类问题中，建议设定
# m = sqrt(d),取n_min = 1,回归问题中，一般m=d/3,n_min=5


plt.style.use('ggplot')
plt.rcParams['figure.figsize'] = 10, 6


n_train = 150
n_test = 1000
noise = 0.1



# 创建一个简单的分类器作为基线。为了简单点，只使用数值数据
df = pd.read_csv(
    "https://labfile.oss.aliyuncs.com/courses/1283/telecom_churn.csv")

# 选择特征特性
cols =[]
for i in df.columns:
    if(df[i].dtype == "float64") or (df[i].dtype == "int64"):
        cols.append(i)

# 将数据集分离为输入变量和目标变量
X,y = df[cols].copy(),np.asarray(df["Churn"],dtype='int8')

# 为验证过程进行分层分割
skf = StratifiedKFold(n_splits=5,shuffle=True,random_state=42)

#用默认参数初始化分类器
rfc = RandomForestClassifier(random_state=42,n_jobs=-1,oob_score=True)

results = cross_val_score(rfc,X,y,cv=skf)
print("CV accuracy score:{:.2f}%".format(results.mean()*100))

skf = StratifiedKFold(n_splits=5,shuffle=True,random_state=42)


train_acc = []
test_acc = []
temp_train_acc = []
temp_test_acc = []

# 进行网课搜索，加速执行时间
trees_grid = [5, 10, 15, 20, 30, 50, 75, 100]

for ntrees in trees_grid:
    # 可以通过设置参数max_depth=max_depth来限制最多的层数，达到正则化的目的
    # 另一个值得调整的重要参数是 min_samples_leaf
    # 考虑 max_features 这一参数
    # 可以通过使用GridSearchCV这个函数来搜索最佳的参数
    #rfc = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1,
    #                             oob_score=True, max_features=100,min_samples_leaf=10,max_depth=20)
    rfc = RandomForestClassifier(
        n_estimators=ntrees,random_state=42,n_jobs=1,oob_score=True)
    temp_train_acc = []
    temp_test_acc = []
    for train_index,test_index in skf.split(X,y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y[train_index], y[test_index]
        rfc.fit(X_train,y_train)
        temp_train_acc.append(rfc.score(X_train,y_train))
        temp_test_acc.append(rfc.score(X_test,y_test))
    train_acc.append(temp_train_acc)
    test_acc.append(temp_test_acc)

# 打印结果。
train_acc,test_acc = np.asarray(train_acc),np.asarray(test_acc)
print("Best accuracy on CV is {:.2f}% with {} trees".format(max(test_acc.mean(axis=1))*100,
                                                            trees_grid[np.argmax(test_acc.mean(axis=1))]))
plt.style.use('ggplot')

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(trees_grid, train_acc.mean(axis=1),
        alpha=0.5, color='blue', label='train')
ax.plot(trees_grid, test_acc.mean(axis=1), alpha=0.5, color='red', label='cv')
ax.fill_between(trees_grid, test_acc.mean(axis=1) - test_acc.std(axis=1),
                test_acc.mean(axis=1) + test_acc.std(axis=1), color='#888888', alpha=0.4)
ax.fill_between(trees_grid, test_acc.mean(axis=1) - 2*test_acc.std(axis=1),
                test_acc.mean(axis=1) + 2*test_acc.std(axis=1), color='#888888', alpha=0.2)
ax.legend(loc='best')
ax.set_ylim([0.88, 1.02])
ax.set_ylabel("Accuracy")
ax.set_xlabel("N_estimators")

# 原实验搜索参数
# parameters = {'max_features': [4, 7, 10, 13], 'min_samples_leaf': [
#     1, 3, 5, 7], 'max_depth': [5, 10, 15, 20]}
# 为加速线上环境执行，优化搜索参数
parameters = {'max_features': [10, 13], 'min_samples_leaf': [1, 3], 'max_depth': [5, 10]}
rfc = RandomForestClassifier(n_estimators=100, random_state=42,
                             n_jobs=-1, oob_score=True)
gcv = GridSearchCV(rfc, parameters, n_jobs=-1, cv=skf, verbose=1)
gcv.fit(X, y)

print(gcv.best_estimator_)
print(gcv.best_score_)
