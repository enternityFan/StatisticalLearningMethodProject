# @Time : 2022-03-09 16:21
# @Author : Phalange
# @File : viewTask.py
# @Software: PyCharm
# C'est la vie,enjoy it! :D

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker
from matplotlib import rcParams
import warnings
warnings.filterwarnings('ignore')
sns.set()
sns.set_context(
    "notebook",
    font_scale=1.5,
    rc={
        "figure.figsize": (11, 8),
        "axes.titlesize": 18
    }
)

rcParams['figure.figsize'] = 11, 8
df = pd.read_csv(
    'https://labfile.oss.aliyuncs.com/courses/1283/mlbootcamp5_train.csv', sep=';')
print('Dataset size: ', df.shape)
print(df.head())


# 分析身高
print('gender == 2:' + str(df[df['gender']==2]['height'].mean()))
print('gender == 2' + str(len(df[df['gender']==2])))
print('gender == 1:' + str(df[df['gender']==1]['height'].mean()))
print('gender == 1:' + str(len(df[df['gender']==1])))
print("所以应该有24470个男性，45530个女性,2为男性，1为女性")
male_height =df[df['gender']==2]['height']
female_height = df[df['gender']==1]['height']
# 画密度图
male_height.plot(kind='density',
                  sharex=False,figsize=(10,4),legend=False,title = 'male_height')
#plt.show()
female_height.plot(kind='density',
                  sharex=False,figsize=(10,4),legend=False,title = 'female_height')
#plt.show()


# 查看男性还是女性喝酒的频率更高
# 频率表
print("男性的比例：")
print(df[df['gender']==2]['alco'].value_counts(normalize=True))
print("女性的比例：")
print(df[df['gender']==1]['alco'].value_counts(normalize=True))

# 问题：数据集中男性和女性吸烟者所占百分比的差值是多少？
m_smoke =df[df['gender']==2]['smoke'].value_counts(normalize=True)
fem_smoke = df[df['gender']==1]['smoke'].value_counts(normalize=True)
print("男性的比例：")
print(m_smoke)
print("女性的比例：")
print(fem_smoke)
print("抽烟者所占比例的差值：")
print(m_smoke[1] - fem_smoke[1])

#  问题：数据集中吸烟者和非吸烟者的年龄中位数之间的差值（以月计）近似是多少？你需要尝试确定出数据集中 age 合理的表示单位。
# 本次挑战规定 1 年为 365.25 天。

df['age_month'] =df['age']/30.4375  #365.25*12 # 目前df['age']的单位是月
df['age_month'] = df['age_month'].astype('int')
smoke_age_mid = df[df['smoke']==1]['age_month'].median()
nosmoke_age_mid = df[df['smoke']==0]['age_month'].median()
smoke_age_mid,nosmoke_age_mid
print("抽烟者与非抽烟者的差值为(月）:"+str(smoke_age_mid - nosmoke_age_mid))

# 问题：计算  [60,65)  年龄区间下，较健康人群（胆固醇类别 1，收缩压低于 120）与高风险人群（胆固醇类别为 3，收缩压  [160,180) ）各自心血管病患所占比例。并最终求得二者比例的近似倍数。


df['years'] = df['age'] / 365.25
df['years'] = df['years'].astype('int')
df['years'][:10]
old_people = df[(df['years'] >=60) & (df['years'] < 65)]
risky_old_people = old_people[(old_people['cholesterol']==3) & (old_people['ap_hi']>=160) & (old_people['ap_hi'] <180)]
health_old_people = old_people[(old_people['cholesterol']==1) &  (old_people['ap_hi'] <120)]
print("较健康人群的患病的概率："+str(health_old_people['cardio'].value_counts(normalize=True)[1]))
print("不健康人群的患病的概率："+str(risky_old_people['cardio'].value_counts(normalize=True)[1]))

# BMI指数分析 BMI = weight / height^2
"""
 问题：请选择下面叙述正确的有：
正常 BMI 指数一般在 18.5 到 25 之间。
[ A ] 数据集样本中 BMI 中位数在正常范围内。
[ B ] 女性的平均 BMI 指数高于男性。
[ C ] 健康人群的 BMI 平均高于患病人群。
[ D ] 健康和不饮酒男性中，BMI 比健康不饮酒女性更接近正常值。
"""
# A
df['BMI'] = df['weight'] / ((df['height'] / 100)** 2)
print("A选项错误，BMI的中位数为：" + str(df['BMI'].median()))

# B
print("女性的BMI的中位数为：" + str(df[df['gender']==1]['BMI'].median()))
print("男性的BMI的中位数为：" + str(df[df['gender']==2]['BMI'].median()))
print("B选项正确")

# C
print("健康人群的BMI的平均值为" + str(df[df['cardio']==0]['BMI'].mean()))
print("患病人群的BMI的平均值为" + str(df[df['cardio']==1]['BMI'].mean()))
print("C选项正确")

# D
print("健康不饮酒的男性的BMI的平均值为" + str(df[(df['cardio']==0) & (df['gender'] == 2) & (df['alco']==0)]['BMI'].mean()))
print("健康不饮酒的女性的BMI的平均值为" + str(df[(df['cardio']==0) & (df['gender'] == 1) & (df['alco']==0)]['BMI'].mean()))
print("D选项正确")


"""
清洗数据
问题：请按照以下列举的项目，过滤掉数据中统计有误的部分：

血压特征中，舒张压高于收缩压的样本。
身高特征中，低于 2.5％ 分位数的样本。
身高特征中，高于 97.5％ 分位数的样本。
体重特征中，低于 2.5％ 分位数的样本。
体重特征中，高于 97.5％ 分位数的样本。
百分位数请使用 pd.Series.quantile 方法进行确定，不熟悉可以阅读  官方文档
"""
filtered_df = df[(df['ap_lo'] <= df['ap_hi']) &
                 (df['height'] >= df['height'].quantile(0.025)) &
                 (df['height'] <= df['height'].quantile(0.975)) &
                 (df['weight'] >= df['weight'].quantile(0.025)) &
                 (df['weight'] <= df['weight'].quantile(0.975))]

# 问题：清洗掉的数据占原数据总量的近似百分比？
print("清洗掉的数据占原数据总量的近似百分比:" + str(-(filtered_df.shape[0] - df.shape[0])/df.shape[0]))


# 问题：使用 heatmap() 绘制特征之间的皮尔逊相关性系数矩阵。 官方答案
df = filtered_df.copy()

corr = df.corr(method='pearson')

# 创建一个 Mask 来隐藏相关矩阵的上三角形
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# 绘制图像
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corr, mask=mask, vmax=1, center=0, annot=True, fmt='.1f',
            square=True, linewidths=.5, cbar_kws={"shrink": .5})
plt.show()

#  问题：绘制身高和性别之间的小提琴图 violinplot()。
# 这里建议通过 hue 参数按性别划分，并通过 scale 参数来计算性别对应的具体数量  官方答案
df_melt = pd.melt(frame=df, value_vars=['height'], id_vars=['gender'])
plt.figure(figsize=(12, 10))
ax = sns.violinplot(
    x='variable',
    y='value',
    hue='gender',
    palette="muted",
    split=True,
    data=df_melt,
    scale='count',
    scale_hue=False
)
#plt.show()



# 下面的不整了，太偏了。