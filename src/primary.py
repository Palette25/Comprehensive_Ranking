from sklearn.model_selection import RandomizedSearchCV
from sklearn import preprocessing
from xgboost.sklearn import XGBClassifier

import xgboost as xgb
import pandas as pd
import numpy as np
import warnings
import math
import copy
import gc

# Define source train data and label dirs and read them into memory
warnings.filterwarnings('ignore')
feature_path ='./feature/'
res_path = '../result/'
data_path = '../dataSet/Primary/'

transaction_df = pd.read_csv(data_path + 'transaction_train_new.csv')
operation_df =  pd.read_csv(data_path + 'operation_train_new.csv')
label = pd.read_csv(data_path + 'tag_train_new.csv')

transaction_test = pd.read_csv(data_path + 'transaction_round1_new.csv')
operation_test = pd.read_csv(data_path + 'operation_round1_new.csv')
sample = pd.read_csv(data_path + 'sample.csv')

# Merge Count of the same name on target value
def merge_count(df1, df2, columns, value, cname):
	add = df1.groupby(columns)[value].count().reset_index().rename(columns = {value:cname})
	df2 = df2.merge(add,on=columns,how="left")
	del add
	gc.collect()
	return df2

# Merge and Calculate different kinds number on target value
def merge_nunique(df1, df2, columns, value, cname):
	add = df1.groupby(columns)[value].nunique().reset_index().rename(columns = {value:cname})
	df2 = df2.merge(add,on=columns,how="left")
	del add
	gc.collect()
	return df2

def merge_value_count(df1, df2, col, value):
	tmp = df1.groupby(col)[value].count().reset_index().rename(columns = {value:'cnt'})
	df = tmp.pivot(index=col[0],columns=col[1],values='cnt').reset_index()
	cname = [col[0]]
	for index in range(1,len(df.columns)):
		cname.append(str(col[1])+'_'+str(df.columns[index]))
	df.columns = cname
	df = df.fillna(0)
	df2 = df2.merge(df,on=str(col[0]),how='left')
	del df,df1
	gc.collect()
	return df2

# Get operation features
def get_op_fea(operation_df):
	# Operation -- Day
	op_fea = operation_df[['UID']].drop_duplicates()
	tmp = operation_df.groupby('UID')['day'].agg([max,min,np.mean]).reset_index()
	tmp.columns = ['UID','op_day_max','op_day_min','op_day_mean']
	op_fea = pd.merge(op_fea,tmp,on='UID',how='left')
	# Operation -- Mode Count
	op_fea = merge_count(operation_df,op_fea,'UID','mode','op_cnt')
	op_fea = merge_nunique(operation_df,op_fea,'UID','mode','op_mode_nunique')
	# Success Count
	op_fea = merge_count(operation_df[operation_df.success==0],op_fea,'UID','mode','op_fail_cnt')
	op_fea = merge_count(operation_df[operation_df.success==1],op_fea,'UID','mode','op_success_cnt')
	op_fea['op_fail_cnt'] = op_fea['op_fail_cnt'].fillna(0)
	op_fea['op_success_cnt'] = op_fea['op_success_cnt'].fillna(0)
	# Operation -- Time
	operation_df['op_hour'] = operation_df['time'].apply(lambda x:int(x.split(':')[0]))
	tmp = operation_df.groupby('UID')['op_hour'].agg([max,min,np.mean]).reset_index()
	tmp.columns=['UID','op_hour_max','op_hour_min','op_hour_mean']
	op_fea = pd.merge(op_fea,tmp,on='UID',how='left')
	# Operation -- OS
	for col in ['os','version','device1','device2','device_code1','device_code2','mac1','ip1','ip2','device_code3','mac2','wifi','geo_code','ip1_sub','ip2_sub']:
		op_fea = merge_nunique(operation_df,op_fea,'UID',col,'op_'+col+'_nunique')
	return op_fea

# Get transaction features
def get_trans_fea(transaction_df):
	trans_fea = transaction_df[['UID']].drop_duplicates()
	# Transaction Channel
	trans_fea = merge_value_count(transaction_df,trans_fea,['UID','channel'],'day')
	trans_fea = merge_count(transaction_df,trans_fea,'UID','channel','trans_cnt')
	trans_fea = merge_nunique(transaction_df,trans_fea,'UID','channel','trans_channel_nunique')

	for col in ['trans_type2','market_type']:
		trans_fea = merge_value_count(transaction_df,trans_fea,['UID',col],'day')
		trans_fea = merge_nunique(transaction_df,trans_fea,'UID',col,'trans_'+col+'_nunique')
	for col in ['trans_type1','merchant','code1','code2','acc_id1','device_code1','device_code2','device_code3','device1','device2','mac1','ip1','acc_id2','acc_id3','geo_code','market_code','ip1_sub']:
		trans_fea = merge_nunique(transaction_df,trans_fea,'UID',col,'trans_'+col+'_nunique')
	return trans_fea

# Set XgBoost Config and Parameters
config = {
	'rounds': 10000,
	'folds': 5
}

params = {
	'booster':'gbtree',
	'objective':'binary:logistic',
	'stratified':True,
	'learning_rate':0.01,
	'min_child_weight':1,
	'max_depth':3,
	'gamma':0,
	'subsample':0.5,
	'colsample_bytree':0.5, 
	'lambda':0.01, 
	'alpha':0.01,
	'eta':0.01,
	'seed':20,
	'silent':1,
	'eval_metric':'auc',
}

# Calculate Custom scores
def CustomScore(preds, dtrain):
	label = dtrain.get_label()
	d = pd.DataFrame()
	d['prob'] = list(preds)
	d['y'] = list(label)
	d = d.sort_values(['prob'], ascending=[0])
	y = d.y
	PosAll = pd.Series(y).value_counts()[1]
	NegAll = pd.Series(y).value_counts()[0]
	
	pCumsum = d['y'].cumsum()
	nCumsum = np.arange(len(y)) - pCumsum + 1
	pCumsumPer = pCumsum / PosAll
	nCumsumPer = nCumsum / NegAll
	
	TR1 = pCumsumPer[abs(nCumsumPer - 0.001).idxmin()]
	TR2 = pCumsumPer[abs(nCumsumPer - 0.005).idxmin()]
	TR3 = pCumsumPer[abs(nCumsumPer - 0.01).idxmin()]
	score = 0.4 * TR1 + 0.3 * TR2 + 0.3 * TR3
	return 'SCORE',float(score)

# Do XgBoost CV
def xgbCV(trainFeature, trainLabel, params, rounds):
	dtrain = xgb.DMatrix(trainFeature, label = trainLabel)
	params['scale_pos_weights '] = (float)(len(trainLabel[trainLabel == 0]))/(float)(len(trainLabel[trainLabel == 1]))
	print ('run cv: ' + 'round: ' + str(rounds))
	res = xgb.cv(params, dtrain, rounds, verbose_eval=100, early_stopping_rounds=200, nfold=3, feval=CustomScore)
	return res

# XgBoost Prediction
def xgbPredict(trainFeature, trainLabel, testFeature, rounds, params):
	params['scale_pos_weights '] = (float)(len(trainLabel[trainLabel == 0]))/len(trainLabel[trainLabel == 1])
	dtrain = xgb.DMatrix(trainFeature.values, label=trainLabel)
	dtest = xgb.DMatrix(testFeature.values)
	watchlist  = [(dtrain,'train')]
	
	model = xgb.train(params, dtrain, rounds, watchlist, verbose_eval=100, feval=CustomScore)
	predict = model.predict(dtest)
	return model,predict

def xgbTrain(params, trainFeature, trainLabel, rounds):
	params['scale_pos_weights '] = (float)(len(trainLabel[trainLabel == 0]))/len(trainLabel[trainLabel == 1])
	dtrain = xgb.DMatrix(trainFeature.values[:int(len(trainFeature.values)/2)], label=trainLabel[0:int(len(trainLabel)/2)])
	dtest = xgb.DMatrix(trainFeature.values[int(len(trainFeature.values)/2):])
	watchlist  = [(dtrain,'train')]
	
	model = xgb.train(params, dtrain, rounds, watchlist, verbose_eval=100, feval=CustomScore)
	predict = model.predict(dtest)

	result = 0
	trainLabel_half = trainLabel[int(len(trainFeature.values)/2):]
	for i in range(len(trainLabel_half)):
		if (predict[i] >= 0.5 and trainLabel[i] == 1) or (predict[i] < 0.5 and trainLabel[i] == 0):
			result += 1

	return float(result) / len(trainLabel)


if __name__ == '__main__':
	op_fea = get_op_fea(operation_df)
	trans_fea = get_trans_fea(transaction_df)

	op_fea_test = get_op_fea(operation_test)
	trans_fea_test = get_trans_fea(transaction_test)

	all_fea = trans_fea.merge(op_fea,on='UID',how='outer')
	trainData = all_fea.merge(label,on='UID',how='left')
	trainFeature = trainData.drop(['Tag','UID'],axis=1)
	trainLabel = trainData.Tag
	#cv_res = xgbCV(trainFeature, trainLabel, params, 10000)
	#cv_res['test-SCORE-mean'][-1:]

	all_fea_test = trans_fea_test.merge(op_fea_test,on='UID',how='outer')
	testFeature = sample[['UID']].merge(all_fea_test,on='UID',how='left')
	sub_id = testFeature['UID']
	testFeature =testFeature.drop('UID',axis=1)

	# Optimize the parameters
	param_op = {
		'learning_rate': [i/100.0 for i in range(10)]
	}

	xgbl = XGBClassifier(
		learning_rate=0.1,
		n_estimators=1000,
		max_depth=5,
		min_child_weight=1,
		gamma=0,
		subsample=0.8,
		colsample_bytree=0.8,
		objective= 'binary:logistic',
		nthread=4,
		scale_pos_weight=1,
		seed=27,
		n_jobs=4
	)

	rand = RandomizedSearchCV(xgbl, param_op, cv=10, scoring='accuracy', random_state=5, iid=False, verbose=10)
	rand.fit(trainFeature, trainLabel)

	print(rand.best_score_)
	print(rand.best_params_)

	# Make result
	'''
	model, predict = xgbPredict(trainFeature, trainLabel, testFeature, 3000, params)

	sub = pd.DataFrame()
	sub['UID'] = sub_id
	sub['Tag'] = predict
	sub.head()

	sub.to_csv(res_path+'baseline.csv',index=0)
	'''