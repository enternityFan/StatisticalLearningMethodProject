# @Time : 2022-03-09 15:33
# @Author : Phalange
# @File : pyScript.py
# @Software: PyCharm
# C'est la vie,enjoy it! :D

import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()
warnings.filterwarnings('ignore')

df = pd.read_csv(
    'https://labfile.oss.aliyuncs.com/courses/1283/telecom_churn.csv')

print(df.head())

features = ['Total day minutes','Total intl calls']

df[features].hist(figsize=(10,4)) # 画条形图
plt.show()

# 画密度图
df[features].plot(kind='density',subplots=True,layout=(1,2),
                  sharex=False,figsize=(10,4),legend=False,title = features)
plt.show()

# 观测数值变量的分布
sns.displot(df['Total intl calls'])
plt.show()

# 箱型图
sns.boxenplot(x='Total intl calls',data=df)
plt.show()

# 提琴形图
_, axes = plt.subplots(1, 2, sharey=True, figsize=(6, 4))
sns.boxplot(data=df['Total intl calls'], ax=axes[0])
sns.violinplot(data=df['Total intl calls'], ax=axes[1])
#plt.show()


# 频率表
print(df['Churn'].value_counts())

# 条形图，是对频率表的图形化表示方法
_, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))

sns.countplot(x='Churn', data=df, ax=axes[0])
sns.countplot(x='Customer service calls', data=df, ax=axes[1])
plt.show()

# 计算特征之间的相关性,这里丢弃了非数值变量
numerical = list(set(df.columns) -
                 set(['State','International plan','Voice mail plan',
                      'Area code','Churn','Customer service calls']))

# 计算和绘图
corr_matrix = df[numerical].corr()
sns.heatmap(corr_matrix)
plt.show()
# 通过热力图，可以去除其他的一些因变量，通过上面的方式

# 散点图
plt.scatter(df['Total day minutes'],df['Total night minutes'])
plt.show()

# 同时绘制直方图的散点图
sns.jointplot(x='Total day minutes', y='Total night minutes',
              data=df, kind='scatter')
#plt.show()
# 绘制平滑过的散点直方图
sns.jointplot('Total day minutes', 'Total night minutes', data=df,
              kind="kde", color="g")
#plt.show()

# 散点图矩阵
#%config InlineBackend.figure_format = 'png'
#sns.pairplot(df[numerical])

# 查看输入变量与目标变量的关系，hue参数制定的是感兴趣的类别特征
sns.lmplot('Total day minutes','Total night minutes',
           data=df,hue='Churn',fit_reg=False)
plt.show()

# 箱型图可视化忠实用户与离网客户这两个互斥分组中数值变量分布的统计数据
numerical.append('Customer service calls')
fig,axes = plt.subplots(nrows=3,ncols=4,figsize=(10,7))
for idx,feat in enumerate(numerical):
    ax = axes[int(idx / 3),idx % 3]
    sns.boxplot(x='Churn',y=feat,data=df,ax=ax)
    ax.set_xlabel("")
    ax.set_ylabel(feat)
fig.tight_layout()
#plt.show()


# 交叉表：通过使用表格的形式表示多个类别变量的频率分布，通过它可以查看某一列或某一行以了解某个变量在另一个变量的作用下的分布情况
print(pd.crosstab(df['State'],df['Churn']).T)

# groupby()计算概率
print(df.groupby(['State'])['Churn'].agg([np.mean]).sort_values(by='mean',ascending=False).T)



# 全局数据集可视化，需要降维的操作

from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

# .map方法把YES、NO转换为数值
X = df.drop(['Churn', 'State'], axis=1)
X['International plan'] = X['International plan'].map({'Yes': 1, 'No': 0})
X['Voice mail plan'] = X['Voice mail plan'].map({'Yes': 1, 'No': 0})

# 归一化数据
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# t-SNE表示
tsne = TSNE(random_state=17)
tsne_repr = tsne.fit_transform(X_scaled)
#可视化出来
plt.scatter(tsne_repr[:,0],tsne_repr[:,1],alpha=5)
plt.show()

# 加上色彩，美化美化
plt.scatter(tsne_repr[:, 0], tsne_repr[:, 1],
            c=df['Churn'].map({False: 'blue', True: 'orange'}))
plt.show()


_, axes = plt.subplots(1, 2, sharey=True, figsize=(12, 5))

for i, name in enumerate(['International plan', 'Voice mail plan']):
    axes[i].scatter(tsne_repr[:, 0], tsne_repr[:, 1],
                    c=df[name].map({'Yes': 'orange', 'No': 'blue'}), alpha=.5)
    axes[i].set_title(name)