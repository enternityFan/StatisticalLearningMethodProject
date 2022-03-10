# @Time : 2022-03-10 20:40
# @Author : Phalange
# @File : sarcasmTest.py
# @Software: PyCharm
# C'est la vie,enjoy it! :D
import eli5
import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import warnings
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_files
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from scipy.sparse import hstack
warnings.filterwarnings('ignore')
# http://labfile.oss.aliyuncs.com/courses/1283/train-balanced-sarcasm.csv.zip

train_df = pd.read_csv('train-balanced-sarcasm.csv')
print(train_df.head())

# 查看数据集变量类别信息
# comment 的数量小于其他特征数量，说明存在缺失值。这里直接将这些缺失数据样本删除。
print(train_df.info())
train_df.dropna(subset=['comment'],inplace=True)
# 查看数据标签类别是否平衡
train_df['label'].value_counts()
# 切分数据为训练和测试集
train_texts, valid_texts, y_train, y_valid = \
    train_test_split(train_df['comment'], train_df['label'], random_state=17)

# 数据可视化探索
# 条形图看讽刺和正常文本长度，用np.lo1p来对数据进行平滑处理
train_df.loc[train_df['label']==1,'comment'].str.len().apply(
    np.log1p).hist(label='sarcastic',alpha=.5)
train_df.loc[train_df['label']==0,'comment'].str.len().apply(
    np.log1p).hist(label='normal',alpha=.5)
plt.legend()
# plt.show()

# np.size看不同子版块评论的总数，mean代表所占比例
sub_df = train_df.groupby('subreddit')['label'].agg([np.size,np.mean,np.sum])
sub_df.sort_values(by='sum',ascending=False).head(10)


# 训练分类模型
# tf-idf 提取文本特征
tf_idf = TfidfVectorizer(ngram_range=(1,2),max_features=50000,min_df=2)
# 建立逻辑回归模型
logit = LogisticRegression(C=1,n_jobs=4,solver='lbfgs',
                           random_state=17,verbose=1)
# 使用 sklearn pipeline 封装2个步骤
tfidf_logit_pipeline = Pipeline([('tf_idf',tf_idf),
                                 ('logit',logit)])

#绘制混淆矩阵
def plot_confusion_matrix(actual, predicted, classes,
                          normalize=False,
                          title='Confusion matrix', figsize=(7, 7),
                          cmap=plt.cm.Blues, path_to_save_fig=None):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    import itertools
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(actual, predicted).T
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=figsize)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('Predicted label')
    plt.xlabel('True label')

    if path_to_save_fig:
        plt.savefig(path_to_save_fig, dpi=300, bbox_inches='tight')

# 开始训练分类模型
# train_texts, valid_texts, y_train, y_valid
tfidf_logit_pipeline.fit(train_texts,y_train)

# 测试效果
print(round(tfidf_logit_pipeline.score(train_texts,y_train),3))
print(round(tfidf_logit_pipeline.score(valid_texts,y_valid),3))
valid_preds = tfidf_logit_pipeline.predict(valid_texts)
plot_confusion_matrix(y_valid, valid_preds,
                      tfidf_logit_pipeline.named_steps['logit'].classes_, figsize=(8, 8))
plt.show()

eli5.show_weights(estimator=tfidf_logit_pipeline.named_steps['logit'],
                  vec=tfidf_logit_pipeline.named_steps['tf_idf'])

# 模型改进 多使用一个特征
subreddits = train_df['subreddit']
train_subreddits,valid_subreddits = train_test_split(
    subreddits,random_state=17)

tf_idf_texts = TfidfVectorizer(
    ngram_range=(1,2),max_features=50000,min_df=2)
tf_idf_subreddits = TfidfVectorizer(ngram_range=(1,1))

X_train_texts = tf_idf_texts.fit_transform(train_texts)
X_valid_texts = tf_idf_texts.transform(valid_texts)
X_train_subreddits = tf_idf_subreddits.fit_transform(train_subreddits)
X_valid_subreddits = tf_idf_subreddits.transform(valid_subreddits)
X_train = hstack([X_train_texts, X_train_subreddits])
X_valid = hstack([X_valid_texts, X_valid_subreddits])

logit.fit(X_train, y_train)
valid_pred = logit.predict(X_valid)
print(accuracy_score(y_valid, valid_pred))