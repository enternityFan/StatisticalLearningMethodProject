# @Time : 2022-03-10 11:00
# @Author : Phalange
# @File : Task.py
# @Software: PyCharm
# C'est la vie,enjoy it! :D

import warnings
import pydotplus
from io import StringIO
from IPython.display import SVG
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV, cross_val_score
import collections
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

plt.rcParams['figure.figsize'] = (10, 8)
warnings.filterwarnings('ignore')

"""
创建一个示例数据集，该数据集表示了 A 会不会和 B 进行第二次约会。
而数据集中的特征包括：外貌，口才，酒精消费，以及第一次约会花了多少钱。
"""
# 创建示例数据集，并对数据类别进行独热编码
def create_df(dic, feature_list):
    out = pd.DataFrame(dic)
    out = pd.concat([out, pd.get_dummies(out[feature_list])], axis=1)
    out.drop(feature_list, axis=1, inplace=True)
    return out
# 保证独热编码后的特征在训练和测试数据中同时存在


def intersect_features(train, test):
    common_feat = list(set(train.keys()) & set(test.keys()))
    return train[common_feat], test[common_feat]


features = ['Looks', 'Alcoholic_beverage', 'Eloquence', 'Money_spent']
# 训练数据
df_train = {}
df_train['Looks'] = ['handsome', 'handsome', 'handsome', 'repulsive',
                     'repulsive', 'repulsive', 'handsome']
df_train['Alcoholic_beverage'] = [
    'yes', 'yes', 'no', 'no', 'yes', 'yes', 'yes']
df_train['Eloquence'] = ['high', 'low', 'average', 'average', 'low',
                         'high', 'average']
df_train['Money_spent'] = ['lots', 'little', 'lots', 'little', 'lots',
                           'lots', 'lots']
df_train['Will_go'] = LabelEncoder().fit_transform(
    ['+', '-', '+', '-', '-', '+', '+'])

df_train = create_df(df_train, features)
print(df_train)

# 测试数据
df_test = {}
df_test['Looks'] = ['handsome', 'handsome', 'repulsive']
df_test['Alcoholic_beverage'] = ['no', 'yes', 'yes']
df_test['Eloquence'] = ['average', 'high', 'average']
df_test['Money_spent'] = ['lots', 'little', 'lots']
df_test = create_df(df_test, features)
print(df_test)
# 保证独热编码后的特征在训练和测试数据中同时存在
y = df_train['Will_go']
df_train, df_test = intersect_features(train=df_train, test=df_test)
df_train

# 尝试自己在纸上计算一下这个决策树
# 决策树
dt = DecisionTreeClassifier(criterion='entropy', random_state=17)
dt.fit(df_train, y)

# 另一个例子有 9 个蓝色球和 11 个黄色球。如果球是蓝色，则让球的标签是 1，否则为 0。
balls = [1 for i in range(9)] + [0 for i in range(11)]  # 生成数据
# 数据分组
# 8 蓝色 和 5 黄色
balls_left = [1 for i in range(8)] + [0 for i in range(5)]
# 1 蓝色 和 6 黄色
balls_right = [1 for i in range(1)] + [0 for i in range(6)]
# 问题：请根据前面的实验内容实现香农熵计算函数 entropy()。

def entropy(x):
    kinds = {}
    H = 0
    for each in x:
        if each not in kinds:
            kinds[each] = 1
        else:
            kinds[each] +=1
    for each in kinds:
        H +=(kinds[each] / len(x))*np.log2(kinds[each] / len(x))
    return -H
print(entropy(balls_left))








# 信息增益的计算函数
def information_gain(root, left, right):
    gain = entropy(root)
    data_length = len(root)

    gain -=(len(left)/data_length * entropy(left) + len(right) /data_length * entropy(right))

    return gain

print(information_gain(balls,balls_left,balls_right))

"""
构建 Adult 数据集决策树
 UCI Adult 人口收入普查数据集前面已经使用过了，其具有以下一些特征：

Age – 连续数值特征
Workclass – 连续数值特征
fnlwgt – 连续数值特征
Education – 类别特征
Education_Num – 连续数值特征
Martial_Status – 类别特征
Occupation – 类别特征
Relationship – 类别特征
Race – 类别特征
Sex – 类别特征
Capital_Gain – 连续数值特征
Capital_Loss – 连续数值特征
Hours_per_week – 连续数值特征
Country – 类别特征
Target – 收入水平，二元分类目标值

"""
data_train = pd.read_csv(
    'https://labfile.oss.aliyuncs.com/courses/1283/adult_train.csv', sep=';')
data_train.tail()
data_test = pd.read_csv(
    'https://labfile.oss.aliyuncs.com/courses/1283/adult_test.csv', sep=';')
data_test.tail()
# 然后，对数据集进行一些必要的清洗。同时，将目标值转换为 0，1 二元数值。

# 移除测试集中的错误数据
data_test = data_test[(data_test['Target'] == ' >50K.')
                      | (data_test['Target'] == ' <=50K.')]

# 将目标编码为 0 和 1
data_train.loc[data_train['Target'] == ' <=50K', 'Target'] = 0
data_train.loc[data_train['Target'] == ' >50K', 'Target'] = 1

data_test.loc[data_test['Target'] == ' <=50K.', 'Target'] = 0
data_test.loc[data_test['Target'] == ' >50K.', 'Target'] = 1


# 输出测试数据概览表，查看特征和目标值的各项统计指标。
# 查看数据规律
print(data_test.describe(include='all').T)
print(data_train['Target'].value_counts())
# 画图
fig = plt.figure(figsize=(25, 15))
cols = 5
rows = np.ceil(float(data_train.shape[1]) / cols)
for i, column in enumerate(data_train.columns):
    ax = fig.add_subplot(rows, cols, i + 1)
    ax.set_title(column)
    if data_train.dtypes[column] == np.object:
        data_train[column].value_counts().plot(kind="bar", axes=ax)
    else:
        data_train[column].hist(axes=ax)
        plt.xticks(rotation="vertical")
plt.subplots_adjust(hspace=0.7, wspace=0.2)

# 检查数据类型
print(data_train.dtypes)
print(data_test.dtypes)
# 把Age的object类型修复为整数类型
data_test['Age'] = data_test['Age'].astype(int)
# 把测试数据集的浮点类型全部处理为整数类型，以便与训练数据对应
data_test['fnlwgt'] = data_test['fnlwgt'].astype(int)
data_test['Education_Num'] = data_test['Education_Num'].astype(int)
data_test['Capital_Gain'] = data_test['Capital_Gain'].astype(int)
data_test['Capital_Loss'] = data_test['Capital_Loss'].astype(int)
data_test['Hours_per_week'] = data_test['Hours_per_week'].astype(int)
# 从数据集中选择类别和连续特征变量
categorical_columns = [c for c in data_train.columns
                       if data_train[c].dtype.name == 'object']
numerical_columns = [c for c in data_train.columns
                     if data_train[c].dtype.name != 'object']

print('categorical_columns:', categorical_columns)
print('numerical_columns:', numerical_columns)

#然后，对连续特征使用中位数对缺失数据进行填充，而类别特征则使用众数进行填充。
# 填充缺失数据
for c in categorical_columns:
    data_train[c].fillna(data_train[c].mode(), inplace=True)
    data_test[c].fillna(data_train[c].mode(), inplace=True)

for c in numerical_columns:
    data_train[c].fillna(data_train[c].median(), inplace=True)
    data_test[c].fillna(data_train[c].median(), inplace=True)
# 对类别特征进行独热编码，以保证数据集特征全部为数值类型方便后续传入模型。

data_train = pd.concat([data_train[numerical_columns],
                        pd.get_dummies(data_train[categorical_columns])], axis=1)

data_test = pd.concat([data_test[numerical_columns],
                       pd.get_dummies(data_test[categorical_columns])], axis=1)
# set(data_train.columns) - set(data_test.columns)#来查看，是否训练集和测试集包含不相干的特征
# data_train.shape,data_test.shape #也可以查看特征行数
# 独热编码之后发现测试数据中没有 Holland，为了与训练数据对应，这里需要创建零值特征进行补齐。
data_test['Country_Holand-Netherlands'] = 0

# 生成训练数据和测试数据
X_train = data_train.drop(['Target'], axis=1)
y_train = data_train['Target']

X_test = data_test.drop(['Target'], axis=1)
y_test = data_test['Target']

dt_clf = DecisionTreeClassifier(max_depth=3,random_state=17)
dt_clf.fit(X_train,y_train)
print(accuracy_score(y_test,dt_clf.predict(X_test)))

# 建立随机森林模型
rf_clf = RandomForestClassifier(n_estimators=100,random_state=17)
rf_clf.fit(X_train,y_train)
print(accuracy_score(y_test,rf_clf.predict(X_test)))