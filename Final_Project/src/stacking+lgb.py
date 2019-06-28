'''
Primary Solution For HuaWei Digix Competition

Solution: Basic Models Stacking + LightGBM
Author: @Palette25
'''

import gensim
import time
import re
import gc
import os

import matplotlib.pyplot as plt
import lightgbm as lgb
import seaborn as sns
import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec, FastText

from sklearn.decomposition import LatentDirichletAllocation

from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score

from datetime import datetime, timedelta
from scipy.sparse import hstack, vstack
from sklearn import preprocessing

from tqdm import tqdm

dataPath = '../dataSet/'

# Step One. Read dataSet and do processing prepare works
train = pd.read_csv(dataPath+'age_train.csv', names=['uid','age'])
test = pd.read_csv(dataPath+'age_test.csv', names=['uid'])
basic_info = pd.read_csv(dataPath+'user_basic_info.csv', 
						names=['uid','gender','city','phone_type','ram','ram_left','rom','rom_left','color','fontSize','ct','carrier','os'])

app_package = pd.read_csv(dataPath+'user_app_actived.csv', names=['uid','appid'])
app_info = pd.read_csv(dataPath+'app_info.csv', names=['appid','category'])

def get_app_str(df_input):
	result = ''
	for ele in df_input.split('#'):
		result += ele + ' '

	return result

app_package['appstr'] = app_package['appid'].apply(lambda x : get_app_str(x), 1)

# Do Tf-Idf Preparation
tfidf = CountVectorizer()

train_str_app = pd.merge(train[['uid']], app_package[['uid','appstr']],on='uid',how='left')
test_str_app = pd.merge(test[['uid']], app_package[['uid','appstr']],on='uid',how='left')
app_package['appstr'] = tfidf.fit_transform(app_package['appstr'])
train_app = tfidf.transform(list(train_str_app['appstr'])).tocsr()
test_app = tfidf.transform(list(test_str_app['appstr'])).tocsr()

all_id = pd.concat([train[['uid']], test[['uid']]])

all_id.index = range(len(all_id))


# Step Two. Do simple models stacking
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

train_feature = train_app
test_feature = test_app
n_folds = 5

df_stack = pd.DataFrame()
df_stack['uid'] = all_id['uid']

labels = train['age']

print('LR Stacking')
stack_train = np.zeros((len(train), 1))
stack_test = np.zeros((len(test), 1))

score_va = 0

kfold = StratifiedKFold(n_splits=n_folds, random_state=100, shuffle=True)

for i, (tr, va) in enumerate(kfold.split(labels)):
	print('LR Stacking: %d/%d' % ((i+1), n_folds))
	clf = LogisticRegression(random_state=100, C=8)
	clf.fit(train_feature[tr], labels[tr])
	score_va = clf.predict_proba(train_feature[va])[:,1]

	score_te = clf.predict_proba(test_feature)[:,1]
	print('Score: ' + str(mean_squared_error(score[va], clf.predict(train_feature[va]))))
	stack_train[va, 0] = score_va
	stack_test[:, 0] += score_te

stack_test /= n_folds
stack = np.vstack([stack_train, stack_test])

df_stack['pack_tfidf_lr_classify_{}'.format('age')] = stack[:, 0]

print(df_stack)