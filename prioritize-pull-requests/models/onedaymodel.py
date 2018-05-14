import pandas as pd
import numpy as np
from sklearn import preprocessing  
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
from sklearn.metrics import auc
from scipy import sparse
from sklearn import metrics
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
import math
from pandas.core.frame import DataFrame

def read_data(name):
	data_path = name
	return pd.read_csv(data_path)


def pre(name):
	data = read_data(name)
	enc = preprocessing.LabelEncoder()
	#print(data['Language'])
	data['Language'] = [str(item) for item in data['Language']]
	enc.fit(data['Language'])
	oneHotLanguage = enc.transform(data['Language'])  
	data['Language'] = oneHotLanguage
	return data



def data_to_vec(data):
	feature_data = [ 'Contain_Fix_Bug', 'Sunday', 'Team_Size', 
	'Commits_Average', 'Rebaseable', 'Private_Repos', 'Comments_Per_Merged_PR', 'Thursday', 
	'Review_Comments_Count', 'Friday', 'Day', 'Contributor', 'Mergeable', 'Language', 'File_Touched_Average',
	 'Contributions', 'User_Accept_Rate', 'Public_Repos', 'Monday', 'Tuesday', 'Forks_Count', 
	 'Closed_Num_Rate', 'Contributor_Num', 'Churn_Average', 'Project_Accept_Rate', 
	 'Deletions', 'Watchers', 'Merge_Latency', 'Saturday', 'Followers', 'Additions', 
	 'Deletions_Per_Week',  'Workload', 'Mergeable_State', 'Assignees_Count',
	 'Wait_Time',  'Comments_Count', 'Comments_Per_Closed_PR', 'Participants_Count', 'Accept_Num',
	 'Following', 'Wednesday', 'Project_Age', 'Prev_PRs', 'Intra_Branch', 'Files_Changed',  'Closed_Num',
	  'Additions_Per_Week', 'Label_Count', 'Open_Issues', 'Organization_Core_Member', 
	'Commits_PR', 'Point_To_IssueOrPR', 'Last_Comment_Mention', 'Stars', 'Close_Latency','X1_0','X1_1','X1_2','X1_3','X1_4'
	,'X1_5','X1_6','X1_7','X1_8','X1_9','X2_0','X2_1','X2_2','X2_3','X2_4','X2_5','X2_6','X2_7','X2_8','X2_9']
	X = data[feature_data]
	return X
	
def data_to_vec_time(data):
	feature_data = [ 'Contain_Fix_Bug', 'Sunday', 'Team_Size', 
	'Commits_Average', 'Rebaseable', 'Private_Repos', 'Comments_Per_Merged_PR', 'Thursday', 
	'Review_Comments_Count', 'Friday', 'Day', 'Contributor', 'Mergeable', 'Language', 'File_Touched_Average',
	 'Contributions', 'User_Accept_Rate', 'Public_Repos', 'Monday', 'Tuesday', 'Forks_Count', 
	 'Closed_Num_Rate', 'Contributor_Num', 'Churn_Average', 'Project_Accept_Rate', 
	 'Deletions', 'Watchers', 'Merge_Latency', 'Saturday', 'Followers', 'Additions', 
	 'Deletions_Per_Week',  'Workload', 'Mergeable_State', 'Assignees_Count',
	 'Comments_Count', 'Comments_Per_Closed_PR', 'Participants_Count', 'Accept_Num',
	 'Following', 'Wednesday', 'Project_Age', 'Prev_PRs', 'Intra_Branch', 'Files_Changed',  'Closed_Num',
	  'Additions_Per_Week', 'Label_Count', 'Open_Issues', 'Organization_Core_Member', 
	'Commits_PR', 'Point_To_IssueOrPR', 'Last_Comment_Mention', 'Stars', 'Close_Latency','X1_0','X1_1','X1_2','X1_3','X1_4'
	,'X1_5','X1_6','X1_7','X1_8','X1_9','X2_0','X2_1','X2_2','X2_3','X2_4','X2_5','X2_6','X2_7','X2_8','X2_9','wait_time_up']
	X = data[feature_data]
	return X
	

def train_and_test_to_vec(test_data):
	X_test= data_to_vec(test_data)
	X_test_time = data_to_vec_time(test_data)
	return X_test,X_test_time


def get_classifiers():
	return {
	'XGBoost':'XGBoost',
	}
	
	
def trained_and_predict_location(clf,X_test,X_test_time):
	if clf == "XGBoost":
		params = {
			'objective': 'binary:logistic',
			'eta': 0.08,
			'colsample_bytree': 0.886,
			'min_child_weight': 1.1,
			'max_depth': 7,
			'subsample': 0.886,
			'gamma': 0.1,
			'lambda':10,
			'verbose_eval': True,
			'eval_metric': 'auc',
			'scale_pos_weight':6,
			'seed': 201703,
			'missing':-1
	   }
		xgbpredict = xgb.Booster(model_file = 'xgb.model')
		xgbpredict_time = xgb.Booster(model_file = 'xgb_time.model')
		print('pridict....')
		xgbtest = xgb.DMatrix(X_test)
		predicted= xgbpredict.predict(xgbtest)
		print('pridict time....')
		xgbtest_time = xgb.DMatrix(X_test_time)
		predicted_time= xgbpredict_time.predict(xgbtest_time)
		return predicted,predicted_time


def traind_and_predict():
	classifiers = get_classifiers()
	for name,clf in classifiers.items():
		for i in range(1,32):
			if i < 10:
				name_file = '2018-01-0' + str(i) + '.csv'
			else:
				name_file = '2018-01-' + str(i) + '.csv'
			test_data = pre(name_file)
			X_test,X_test_time = train_and_test_to_vec(test_data)
			predicted,predicted_time = trained_and_predict_location(clf, X_test,X_test_time)
			result = [math.exp(predicted[index]) + math.exp(predicted_time[index]) for index in range(len(predicted))]
			result = {'result':result}
			result = DataFrame(result)
			data = read_data(name_file)
			result = pd.concat([result,data],axis = 1)
			result['result'] = result['result'].astype('float')
			result=result.sort_values(by=['result'],ascending =False)
			result.to_csv('result_'+ name_file)
			#print(result)


traind_and_predict()

'''
result=[]
for i in range(10):
    result.append(train_test())
print('max:{},min{},avg:{}'.format(max(result),min(result),Get_Average(result)))
'''
