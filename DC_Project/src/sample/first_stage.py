import pandas as pd
import numpy as np
import lightgbm as lgb
import warnings

from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings('ignore')

def tpr_weight_funtion(y_true,y_predict):
	d = pd.DataFrame()
	d['prob'] = list(y_predict)
	d['y'] = list(y_true)
	d = d.sort_values(['prob'], ascending=[0])
	y = d.y
	PosAll = pd.Series(y).value_counts()[1]
	NegAll = pd.Series(y).value_counts()[0]
	pCumsum = d['y'].cumsum()
	nCumsum = np.arange(len(y)) - pCumsum + 1
	pCumsumPer = pCumsum / PosAll
	nCumsumPer = nCumsum / NegAll
	TR1 = pCumsumPer[abs(nCumsumPer-0.001).idxmin()]
	TR2 = pCumsumPer[abs(nCumsumPer-0.005).idxmin()]
	TR3 = pCumsumPer[abs(nCumsumPer-0.01).idxmin()]
	return 'TC_AUC',0.4 * TR1 + 0.3 * TR2 + 0.3 * TR3,True

path = '../../dataSet/Primary/'

op_train = pd.read_csv(path + 'operation_train_new.csv')
trans_train = pd.read_csv(path + 'transaction_train_new.csv')

op_test = pd.read_csv(path + 'operation_round1_new.csv')
trans_test = pd.read_csv(path + 'transaction_round1_new.csv')
y = pd.read_csv(path + 'tag_train_new.csv')
sub = pd.read_csv(path + 'sample.csv')

# Check Device and User Infos
def device_user(df,trans,op):
	trans_device=['ip1','mac1','device1','ip1_sub']
	op_device=['ip1','mac1','wifi','device1','ip1_sub']
	for feature in trans_device:
		trans1=trans.loc[:,['UID',feature]].drop_duplicates()
		trans2=trans1.groupby(feature).size().reset_index(name='trans_'+feature+'_count2')
		trans2=pd.merge(trans1,trans2,on=feature)
		trans2_sum=trans2.groupby(by='UID')['trans_'+feature+'_count2'].sum().reset_index(name='trans_'+feature+'_sum2')
		trans2_nunique2=trans1.groupby('UID')[feature].nunique().reset_index(name='trans_'+feature+'_nunique2')
		trans2_sum=pd.merge(trans2_sum,trans2_nunique2,on='UID')
		trans2_sum['trans_'+feature+'_sum2']=trans2_sum['trans_'+feature+'_sum2']-trans2_sum['trans_'+feature+'_nunique2']
		X=trans2_sum['trans_'+feature+'_sum2'].quantile(0.5)
		trans2_sum['trans_'+feature+'_cl2']=trans2_sum['trans_'+feature+'_sum2']+X

		trans2_max=trans2.groupby(by='UID')['trans_'+feature+'_count2'].max().reset_index(name='trans_'+feature+'_max2')
		trans2_max['trans_'+feature+'_max2']=trans2_max['trans_'+feature+'_max2']-1
		trans2_x=pd.merge(trans2_sum,trans2_max,on='UID')
		trans2_x['trans'+feature+'_cl2_per']=trans2_x['trans_'+feature+'_max2']/trans2_x['trans_'+feature+'_cl2']
		trans2_x=trans2_x.loc[:,['UID','trans_'+feature+'_sum2','trans_'+feature+'_max2','trans_'+feature+'_cl2_per']]
		df=pd.merge(df,trans2_x,on='UID',how='outer') 


	for feature in op_device:
		op1=op.loc[:,['UID',feature]].drop_duplicates()
		op2=op1.groupby(feature).size().reset_index(name='op_'+feature+'_count2')
		op2=pd.merge(op1,op2,on=feature)
		op2_sum=op2.groupby(by='UID')['op_'+feature+'_count2'].sum().reset_index(name='op_'+feature+'_sum2')
		op2_nunique2=op1.groupby('UID')[feature].nunique().reset_index(name='op_'+feature+'_nunique2')
		op2_sum=pd.merge(op2_sum,op2_nunique2,on='UID')
		op2_sum['op_'+feature+'_sum2']=op2_sum['op_'+feature+'_sum2']-op2_sum['op_'+feature+'_nunique2']
		X=op2_sum['op_'+feature+'_sum2'].quantile(0.5)
		op2_sum['op_'+feature+'_cl2']=op2_sum['op_'+feature+'_sum2']+X

		op2_max=op2.groupby(by='UID')['op_'+feature+'_count2'].max().reset_index(name='op_'+feature+'_max2')
		op2_max['op_'+feature+'_max2']=op2_max['op_'+feature+'_max2']-1
		op2_x=pd.merge(op2_sum,op2_max,on='UID')
		op2_x['op'+feature+'_cl2_per']=op2_x['op_'+feature+'_max2']/op2_x['op_'+feature+'_cl2']
		op2_x=op2_x.loc[:,['UID','op_'+feature+'_sum2','op_'+feature+'_max2','op_'+feature+'_cl2_per']]
		df=pd.merge(df,op2_x,on='UID',how='outer')

	return df

# Get Features
def get_feature(op,trans,label):
	for feature in op.columns[:]:
		if feature not in ['day']:
			if feature != 'UID':
				label = label.merge(op.groupby(['UID'])[feature].count().reset_index(),on='UID',how='left')
				label =label.merge(op.groupby(['UID'])[feature].nunique().reset_index(),on='UID',how='left')
			for deliver in ['ip1','mac1','mac2','geo_code']:
				if feature not in deliver:
					if feature != 'UID':
						temp = op[['UID',deliver]].merge(op.groupby([deliver])[feature].count().reset_index(),on=deliver,how='left')[['UID',feature]] 
						temp = temp.groupby('UID')[feature].sum().reset_index()
						temp.columns = ['UID',feature+deliver]
						label =label.merge(temp,on='UID',how='left')
						del temp
						temp = op[['UID',deliver]].merge(op.groupby([deliver])[feature].nunique().reset_index(),on=deliver,how='left')[['UID',feature]] 
						temp = temp.groupby('UID')[feature].sum().reset_index()
						temp.columns = ['UID',feature+deliver]
						label =label.merge(temp,on='UID',how='left')
						del temp
					else:
						temp = op[['UID',deliver]].merge(op.groupby([deliver])[feature].count().reset_index(),on=deliver,how='left')[['UID_x','UID_y']] 
						temp = temp.groupby('UID_x')['UID_y'].sum().reset_index()
						temp.columns = ['UID',feature+deliver]
						label =label.merge(temp,on='UID',how='left')
						del temp
						temp = op[['UID',deliver]].merge(op.groupby([deliver])[feature].nunique().reset_index(),on=deliver,how='left')[['UID_x','UID_y']] 
						temp = temp.groupby('UID_x')['UID_y'].sum().reset_index()
						temp.columns = ['UID',feature+deliver]
						label =label.merge(temp,on='UID',how='left')
						del temp

		else:
			print(feature)
			label =label.merge(op.groupby(['UID'])[feature].count().reset_index(),on='UID',how='left')
			label =label.merge(op.groupby(['UID'])[feature].nunique().reset_index(),on='UID',how='left')
			label =label.merge(op.groupby(['UID'])[feature].max().reset_index(),on='UID',how='left')
			label =label.merge(op.groupby(['UID'])[feature].min().reset_index(),on='UID',how='left')
			label =label.merge(op.groupby(['UID'])[feature].sum().reset_index(),on='UID',how='left')
			label =label.merge(op.groupby(['UID'])[feature].mean().reset_index(),on='UID',how='left')
			label =label.merge(op.groupby(['UID'])[feature].std().reset_index(),on='UID',how='left')

			label =label.merge(op.groupby(['UID'])[feature].median().reset_index(),on='UID',how='left')
			label =label.merge(op.groupby(['UID'])[feature].skew().reset_index(),on='UID',how='left')


			for deliver in ['ip1','mac1','mac2']:
				if feature not in deliver:
					temp = op[['UID',deliver]].merge(op.groupby([deliver])[feature].count().reset_index(),on=deliver,how='left')[['UID',feature]] 
					temp = temp.groupby('UID')[feature].sum().reset_index()
					temp.columns = ['UID',feature+deliver]
					label =label.merge(temp,on='UID',how='left')
					del temp
					temp = op[['UID',deliver]].merge(op.groupby([deliver])[feature].nunique().reset_index(),on=deliver,how='left')[['UID',feature]] 
					temp = temp.groupby('UID')[feature].sum().reset_index()
					temp.columns = ['UID',feature+deliver]
					label =label.merge(temp,on='UID',how='left')
					del temp
					temp = op[['UID',deliver]].merge(op.groupby([deliver])[feature].max().reset_index(),on=deliver,how='left')[['UID',feature]] 
					temp = temp.groupby('UID')[feature].mean().reset_index()
					temp.columns = ['UID',feature+deliver]
					label =label.merge(temp,on='UID',how='left')
					del temp
					temp = op[['UID',deliver]].merge(op.groupby([deliver])[feature].min().reset_index(),on=deliver,how='left')[['UID',feature]] 
					temp = temp.groupby('UID')[feature].mean().reset_index()
					temp.columns = ['UID',feature+deliver]
					label =label.merge(temp,on='UID',how='left')
					del temp
					temp = op[['UID',deliver]].merge(op.groupby([deliver])[feature].sum().reset_index(),on=deliver,how='left')[['UID',feature]] 
					temp = temp.groupby('UID')[feature].mean().reset_index()
					temp.columns = ['UID',feature+deliver]
					label =label.merge(temp,on='UID',how='left')
					del temp
					temp = op[['UID',deliver]].merge(op.groupby([deliver])[feature].mean().reset_index(),on=deliver,how='left')[['UID',feature]] 
					temp = temp.groupby('UID')[feature].mean().reset_index()
					temp.columns = ['UID',feature+deliver]
					label =label.merge(temp,on='UID',how='left')
					del temp
					temp = op[['UID',deliver]].merge(op.groupby([deliver])[feature].std().reset_index(),on=deliver,how='left')[['UID',feature]] 
					temp = temp.groupby('UID')[feature].mean().reset_index()
					temp.columns = ['UID',feature+deliver]
					label =label.merge(temp,on='UID',how='left')
					del temp
					
					
	for feature in trans.columns[1:]:
		if feature not in ['trans_amt','bal','day']:
			if feature != 'UID':
				label =label.merge(trans.groupby(['UID'])[feature].count().reset_index(),on='UID',how='left')
				label =label.merge(trans.groupby(['UID'])[feature].nunique().reset_index(),on='UID',how='left')
			for deliver in ['merchant','ip1','mac1','geo_code',]:
				if feature not in deliver: 
					if feature != 'UID':
						temp = trans[['UID',deliver]].merge(trans.groupby([deliver])[feature].count().reset_index(),on=deliver,how='left')[['UID',feature]] 
						temp = temp.groupby('UID')[feature].sum().reset_index()
						temp.columns = ['UID',feature+deliver]
						label =label.merge(temp,on='UID',how='left')
						del temp
						temp = trans[['UID',deliver]].merge(trans.groupby([deliver])[feature].nunique().reset_index(),on=deliver,how='left')[['UID',feature]] 
						temp = temp.groupby('UID')[feature].sum().reset_index()
						temp.columns = ['UID',feature+deliver]
						label =label.merge(temp,on='UID',how='left')
						del temp
					else:
						temp = trans[['UID',deliver]].merge(trans.groupby([deliver])[feature].count().reset_index(),on=deliver,how='left')[['UID_x','UID_y']] 
						temp = temp.groupby('UID_x')['UID_y'].sum().reset_index()
						temp.columns = ['UID',feature+deliver]
						label =label.merge(temp,on='UID',how='left')
						del temp
						temp = trans[['UID',deliver]].merge(trans.groupby([deliver])[feature].nunique().reset_index(),on=deliver,how='left')[['UID_x','UID_y']] 
						temp = temp.groupby('UID_x')['UID_y'].sum().reset_index()
						temp.columns = ['UID',feature+deliver]
						label =label.merge(temp,on='UID',how='left')
						del temp

		else:
			print(feature)
			label =label.merge(trans.groupby(['UID'])[feature].count().reset_index(),on='UID',how='left')
			label =label.merge(trans.groupby(['UID'])[feature].nunique().reset_index(),on='UID',how='left')
			label =label.merge(trans.groupby(['UID'])[feature].max().reset_index(),on='UID',how='left')
			label =label.merge(trans.groupby(['UID'])[feature].min().reset_index(),on='UID',how='left')
			label =label.merge(trans.groupby(['UID'])[feature].sum().reset_index(),on='UID',how='left')
			label =label.merge(trans.groupby(['UID'])[feature].mean().reset_index(),on='UID',how='left')
			label =label.merge(trans.groupby(['UID'])[feature].std().reset_index(),on='UID',how='left')


			label =label.merge(trans.groupby(['UID'])[feature].median().reset_index(),on='UID',how='left')
			label =label.merge(trans.groupby(['UID'])[feature].skew().reset_index(),on='UID',how='left')

			for deliver in ['merchant','ip1','mac1']:
				if feature not in deliver:
					temp = trans[['UID',deliver]].merge(trans.groupby([deliver])[feature].count().reset_index(),on=deliver,how='left')[['UID',feature]] 
					temp = temp.groupby('UID')[feature].sum().reset_index()
					temp.columns = ['UID',feature+deliver]
					label =label.merge(temp,on='UID',how='left')
					del temp
					temp = trans[['UID',deliver]].merge(trans.groupby([deliver])[feature].nunique().reset_index(),on=deliver,how='left')[['UID',feature]] 
					temp = temp.groupby('UID')[feature].sum().reset_index()
					temp.columns = ['UID',feature+deliver]
					label =label.merge(temp,on='UID',how='left')
					del temp
					temp = trans[['UID',deliver]].merge(trans.groupby([deliver])[feature].max().reset_index(),on=deliver,how='left')[['UID',feature]] 
					temp = temp.groupby('UID')[feature].mean().reset_index()
					temp.columns = ['UID',feature+deliver]
					label =label.merge(temp,on='UID',how='left')
					del temp
					temp = trans[['UID',deliver]].merge(trans.groupby([deliver])[feature].min().reset_index(),on=deliver,how='left')[['UID',feature]] 
					temp = temp.groupby('UID')[feature].mean().reset_index()
					temp.columns = ['UID',feature+deliver]
					label =label.merge(temp,on='UID',how='left')
					del temp
					temp = trans[['UID',deliver]].merge(trans.groupby([deliver])[feature].sum().reset_index(),on=deliver,how='left')[['UID',feature]] 
					temp = temp.groupby('UID')[feature].mean().reset_index()
					temp.columns = ['UID',feature+deliver]
					label =label.merge(temp,on='UID',how='left')
					del temp
					temp = trans[['UID',deliver]].merge(trans.groupby([deliver])[feature].mean().reset_index(),on=deliver,how='left')[['UID',feature]] 
					temp = temp.groupby('UID')[feature].mean().reset_index()
					temp.columns = ['UID',feature+deliver]
					label =label.merge(temp,on='UID',how='left')
					del temp
					temp = trans[['UID',deliver]].merge(trans.groupby([deliver])[feature].std().reset_index(),on=deliver,how='left')[['UID',feature]] 
					temp = temp.groupby('UID')[feature].mean().reset_index()
					temp.columns = ['UID',feature+deliver]
					label =label.merge(temp,on='UID',how='left')
					del temp
					
	print("Get Features Done~")
	return label

# Definition of Tran and Test Datas and Labels
train = get_feature(op_train,trans_train,y).fillna(-1)
test = get_feature(op_test,trans_test,sub).fillna(-1)

train = device_user(train,trans_train,op_train)
test = device_user(test,trans_test,op_test)

train = train.drop(['Tag'],axis = 1).fillna(-1)
label = y['Tag']

test_id = test['UID']
test = test.drop(['Tag'],axis = 1).fillna(-1)

# Define the lgbm Classifier, best parameters decided by bayes optimizer
lgb_model = lgb.LGBMClassifier(boosting_type='gbdt', num_leaves=100, reg_alpha=3, reg_lambda=5, max_depth=-1,
	n_estimators=5000, objective='binary', subsample=0.9, colsample_bytree=0.77, subsample_freq=1, learning_rate=0.05,
	random_state=1000, n_jobs=3, min_child_weight=4, min_child_samples=5, min_split_gain=0)

# K-Fold to check model accuracy
skf = StratifiedKFold(n_splits=5, random_state=2018, shuffle=True)
best_score = []

oof_preds = np.zeros(train.shape[0])
sub_preds = np.zeros(test_id.shape[0])

# Predicting and result labels stroing
for index, (train_index, test_index) in enumerate(skf.split(train, label)):
	lgb_model.fit(train.iloc[train_index], label.iloc[train_index], verbose=50,
				  eval_set=[(train.iloc[train_index], label.iloc[train_index]),
							(train.iloc[test_index], label.iloc[test_index])], early_stopping_rounds=30)
	best_score.append(lgb_model.best_score_['valid_1']['binary_logloss'])
	print(best_score)
	oof_preds[test_index] = lgb_model.predict_proba(train.iloc[test_index], num_iteration=lgb_model.best_iteration_)[:,1]

	test_pred = lgb_model.predict_proba(test, num_iteration=lgb_model.best_iteration_)[:, 1]
	sub_preds += test_pred / 5
	print('test mean:', test_pred.mean())
	predict_result['predicted_score'] = predict_result['predicted_score'] + test_pred


m = tpr_weight_funtion(y_predict=oof_preds,y_true=label)

sub = pd.read_csv(path + 'sample.csv')
sub['Tag'] = sub_preds

res_path = '../../result/'
sub.to_csv(res_path + 'result.csv',index=False)