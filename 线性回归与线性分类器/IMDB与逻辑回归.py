# @Time : 2022-03-10 19:47
# @Author : Phalange
# @File : IMDB与逻辑回归.py
# @Software: PyCharm
# C'est la vie,enjoy it! :D


import os
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_files
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression