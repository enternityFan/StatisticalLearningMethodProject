# @Time : 2022-03-10 19:47
# @Author : Phalange
# @File : IMDB与逻辑回归.py
# @Software: PyCharm
# C'est la vie,enjoy it! :D


import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_files
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV

PATH_TO_IMDB = "aclImdb/"
reviews_train = load_files(os.path.join(PATH_TO_IMDB, "train"),
                           categories=['pos', 'neg'])
text_train, y_train = reviews_train.data, reviews_train.target
reviews_test = load_files(os.path.join(PATH_TO_IMDB, "test"),
                          categories=['pos', 'neg'])
text_test,y_test = reviews_test.data,reviews_test.target
print("Number of documents in training data: %d" % len(text_train))
print(np.bincount(y_train))
print("Number of documents in test data: %d" % len(text_test))
print(np.bincount(y_test))

# 单词的简单计数
cv = CountVectorizer()
cv.fit(text_train)
print("字典的长度为：" + str(len(cv.vocabulary_)))
# 查看创建后的「单词」样本，发现 IMDB 数据集已经自动进行了文本处理（自动化文本处理不在本实验讨论范围，如果感兴趣可以自行搜索）。
print(cv.get_feature_names()[:50])
print(cv.get_feature_names()[50000:50050])

#使用单词的索引编码训练集的句子，用稀疏矩阵保存。
X_train = cv.transform(text_train)

# 把每个单词转换为对应的单词索引
# X_train[19726].nonzero()
X_test = cv.transform(text_test)

logit = LogisticRegression(solver='lbfgs',n_jobs=1,random_state=7)
logit.fit(X_train,y_train)

print(round(logit.score(X_train,y_train),3))
print(round(logit.score(X_test,y_test),3))

# 可视化模型参数
def visualize_coefficients(classifier, feature_names, n_top_features=25):
    # get coefficients with large absolute values
    coef = classifier.coef_.ravel()
    positive_coefficients = np.argsort(coef)[-n_top_features:]
    negative_coefficients = np.argsort(coef)[:n_top_features]
    interesting_coefficients = np.hstack(
        [negative_coefficients, positive_coefficients])
    # plot them
    plt.figure(figsize=(15, 5))
    colors = ["red" if c < 0 else "blue" for c in coef[interesting_coefficients]]
    plt.bar(np.arange(2 * n_top_features),
            coef[interesting_coefficients], color=colors)
    feature_names = np.array(feature_names)
    plt.xticks(np.arange(1, 1 + 2 * n_top_features),
               feature_names[interesting_coefficients], rotation=60, ha="right")

visualize_coefficients(logit, cv.get_feature_names())
plt.show()

# 对逻辑回归的正则化系数进行调参。make_pipeline() 确保的序列顺序，在训练数据上应用 CountVectorizer() 方法，然后训练逻辑回归模型。
text_pipe_logit = make_pipeline(CountVectorizer(),
                                LogisticRegression(solver='lbfgs',n_jobs=1,
                                                   random_state=7))

text_pipe_logit.fit(text_train,y_train)
print(text_pipe_logit.score(text_test,y_test))

# 该单元格执行时间较长，耐心等待一下
param_grid_logit = {'logisticregression__C': np.logspace(-5, 0, 6)[4:5]}
grid_logit = GridSearchCV(text_pipe_logit,
                          param_grid_logit,
                          return_train_score=True,
                          cv=3, n_jobs=1)
# 速度慢，现在我也不用就给关了
#grid_logit.fit(text_train, y_train)

print(grid_logit.best_params_)
print(grid_logit.best_score_)

# 调优后的逻辑回归模型在验证集上的准确率。
print(grid_logit.score(text_test, y_test))

# 可以用随机森林也来分类试试。
