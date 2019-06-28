'''
Perform User-Behavior Feature extract and dimension reduction


Author: @Palette25
'''
import time
import gc

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.decomposition import LatentDirichletAllocation
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from datetime import datetime, timedelta
import lightgbm as lgb

from tqdm import tqdm

app_usage_file_num = 14

behavior_info = pd.read_csv(dataPath+'user_behavior.csv',
							names=['uid','times','A','B','C','D','E','F','G'])

dtime = []
wtime = []
atime = []
dapp = []

for index in range(app_usage_file_num):
	current_usage = pd.read_csv(dataPath+'user_app_usage_' + str(index) + '.csv',
							names=['uid','appid','duration','times','date'])

	current_usage['datetime'] = current_usage['date'].dt.date
	current_usage['dayofweek'] = current_usage['date'].dt.dayofweek

	# Every day user play device's mean time
	dtime_temp = current_usage.groupby(['uid','datetime'])['duration'].agg('sum')
	wtime_temp = current_usage.groupby(['uid', 'dayofweek'])['duration'].agg('sum')
	atime_temp = current_usage.groupby(['uid', 'appid'])['duration'].agg('sum')

	dapp = current_usage[['uid', 'datetime', 'appid']].drop_duplicates().groupby(
			['uid', 'datetime'])['appid'].agg(' '.join)
	dapp.reset_index()


