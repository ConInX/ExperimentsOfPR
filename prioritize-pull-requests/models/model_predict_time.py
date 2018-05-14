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

def read_data():
	data_path = 'update.csv'
	return pd.read_csv(data_path)

def train_test():
	ans = traind_and_predict(train_data,train_label,test_data,test_label)
	return ans
	
def split():
	data = read_data()
	enc = preprocessing.LabelEncoder()
	#print(data['Language'])
	data['Language'] = [str(item) for item in data['Language']]
	enc.fit(data['Language'])
	oneHotLanguage = enc.transform(data['Language'])  
	data['Language'] = oneHotLanguage
	#print(data['Language'][0])
	#return
	for item in data['wait_time_label']:
		if item:
			item = 1
		else:
			item = 0
	#print(data['wait_time_label'])
	data_label = data['wait_time_label']
	train_data,test_data,train_label,test_label = train_test_split(data,data_label,test_size=0.1,stratify=data_label)
	#print(len(train_data))
	#print(len(test_data))
	return train_data,train_label,test_data,test_label
	
def train_test_split_(X,y,test_size=0.3):
	train_size = int((1 - test_size)*X.shape[0])
	if isinstance(X,np.ndarray):
		return X[:train_size],X[train_size:],y[:train_size],y[train_size:]
	else:
		return X.iloc[:train_size],X.iloc[train_size:],y.iloc[:train_size],y.iloc[train_size:]



def data_to_vec(data,label):
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
	'Commits_PR', 'Point_To_IssueOrPR', 'Last_Comment_Mention', 'Stars', 'Close_Latency','wait_time_up']
	y = label
	#data['X1'] = [item[1:-1].split(',') for item in data['X1']]
	#data['X2'] = [item[1:-1].split(',') for item in data['X2']]
	#X = pd.concat([data[feature_data],pd.Series(data['X1']),pd.Series(data['X2'])],axis = 1)
	#X = X['X1']
	X = data[feature_data]
	return X,y

def train_and_test_to_vec(train_data,train_label,test_data,test_label):
	X_train,y_train = data_to_vec(train_data,train_label)
	#print(X_train)
	X_test,y_test = data_to_vec(test_data,test_label)
	#print(y_test)
	return X_train,y_train,X_test,y_test


def get_classifiers():
	return {
	'random forest': RandomForestClassifier(n_jobs=4, n_estimators=200, bootstrap=False, class_weight='balanced'),
	'LinearSVC': LinearSVC(),
	'LogisticRegression': LogisticRegression(),
	'XGBoost':'XGBoost',
	}
def XGBoost():
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
	xgbtrain = xgb.DMatrix(X_train, y_train)
	xgbtest = xgb.DMatrix(X_test)
	model = xgb.train(params, xgbtrain, num_boost_round=200)
	#gc.collect()
	predicted= model.predict(xgbtest)
	return predicted
	
def trained_and_predict_location(clf, X_train, y_train, X_test, y_test):
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
		xgbtrain = xgb.DMatrix(X_train, y_train)
		xgbtest = xgb.DMatrix(X_test)
		print('fitting....')
		model = xgb.train(params, xgbtrain, num_boost_round=200)
		#gc.collect()
		print('pridict....')
		predicted= model.predict(xgbtest)
		return predicted
	else:
		print('fitting....')
		clf = clf.fit(X_train, y_train)
		print('pridict....')
		return clf.predict(X_test)
def Get_Average(list):
	sum = 0
	for item in list:
		sum += item
	return sum/len(list)
def traind_and_predict():
	classifiers = get_classifiers()
	for name,clf in classifiers.items():
		result=[]
		for i in range(10):
			train_data,train_label,test_data,test_label = split()
			X_train,y_train,X_test,y_test = train_and_test_to_vec(train_data,train_label,test_data,test_label)
			predicted = trained_and_predict_location(clf, X_train, y_train, X_test, y_test)
			#score = accuracy_score(y_test, predicted)
			#fpr, tpr, thresholds = metrics.roc_curve(y_test, predicted, pos_label=2)
			#score = auc(fpr, tpr)
			score = roc_auc_score(y_test, predicted)
			result.append(score)
		print(name + "--->"+'max:{},min:{},avg:{}'.format(max(result),min(result),Get_Average(result)))


traind_and_predict()


'''
result=[]
for i in range(10):
    result.append(train_test())
print('max:{},min{},avg:{}'.format(max(result),min(result),Get_Average(result)))
'''