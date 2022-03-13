# @Time : 2022-03-11 9:36
# @Author : Phalange
# @File : Bootst.py
# @Software: PyCharm
# C'est la vie,enjoy it! :D


# Bagging 基于统计学中的 Bootstraping（自助法），该方法令复杂模型的统计评估变得更加可行。
import warnings
import seaborn as sns
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
plt.style.use('ggplot')
plt.rcParams['figure.figsize'] = 10, 6

warnings.filterwarnings('ignore')

# 还是离线率的数据集
telecom_data = pd.read_csv(
    'https://labfile.oss.aliyuncs.com/courses/1283/telecom_churn.csv')

fig = sns.kdeplot(telecom_data[telecom_data['Churn']
                               == False]['Customer service calls'], label='Loyal')
fig = sns.kdeplot(telecom_data[telecom_data['Churn']
                               == True]['Customer service calls'], label='Churn')
fig.set(xlabel='Number of calls', ylabel='Density')
#plt.show()
# 通过图可以看出来，相比那些逐渐离网的客户，忠实客户呼叫客服的次数更少。

# 产生bootstrap样本
def get_bootstrap_samples(data, n_samples):
    indices = np.random.randint(0, len(data), (n_samples, len(data)))
    samples = data[indices]
    return samples


# 产生区间估计
def stat_intervals(stat, alpha):
    boundaries = np.percentile(
        stat, [100 * alpha / 2., 100 * (1 - alpha / 2.)])
    return boundaries

# 分割数据集，分组为忠实客户和离网客户
loyal_calls = telecom_data.loc[telecom_data['Churn'] == False,
                'Customer service calls'].values
churn_calls = telecom_data.loc[telecom_data['Churn'] == True,
                'Customer service calls'].values

# 固定随机数种子
np.random.seed(0)

# 生成样本，并计算各自的均值
loyal_mean_scores = [np.mean(sample) for sample in get_bootstrap_samples(loyal_calls,1000)]

churn_mean_scores = [np.mean(sample) for sample in get_bootstrap_samples(churn_calls,1000)]

# 打印区间估计值，将区间定义为95%
print("Service calls from loyal:mean interval",
      stat_intervals(loyal_mean_scores,0.05))
print("Service calls from churn:mean interval",
      stat_intervals(churn_mean_scores,0.05))
