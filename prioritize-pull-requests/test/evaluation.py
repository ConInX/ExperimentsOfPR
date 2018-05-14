#coding=utf-8  
import math
from pandas.core.frame import DataFrame
import json
import re
import gensim
import csv
import time
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
import os
import sys
# global config
raw_data='filter-raw-data.txt'
accept_data = 'accept_predict.csv'
response_data = 'response_data.csv'


def generate_accept_data(project_full_name, start_time,end_time):
    model = gensim.models.Word2Vec.load('word2vec_model')
    file = open(raw_data,"r")
    fieldnames = []
    for line in file.readlines():
      load_dict = json.loads(line)
      fieldnames = list(load_dict.keys())
      break
    fieldnames.append('X1_0')
    fieldnames.append('X1_1')
    fieldnames.append('X1_2')
    fieldnames.append('X1_3')
    fieldnames.append('X1_4')
    fieldnames.append('X1_5')
    fieldnames.append('X1_6')
    fieldnames.append('X1_7')
    fieldnames.append('X1_8')
    fieldnames.append('X1_9')
    fieldnames.append('X2_0')
    fieldnames.append('X2_1')
    fieldnames.append('X2_2')
    fieldnames.append('X2_3')
    fieldnames.append('X2_4')
    fieldnames.append('X2_5')
    fieldnames.append('X2_6')
    fieldnames.append('X2_7')
    fieldnames.append('X2_8')
    fieldnames.append('X2_9')
    file = open(raw_data,"r")
    with open(accept_data,'w',newline='',errors='ignore') as f:
      f_csv = csv.DictWriter(f,fieldnames=fieldnames)
      f_csv.writeheader()
      for line in file.readlines():
        try:
            load_dict = json.loads(line)
            if load_dict['url'].find(project_full_name) != -1 and time.strptime(str(load_dict['Timeline'][1]['Created_At'])[2:-2],"%Y-%m-%dT%H:%M:%SZ") >= start_time:
              continue
            if load_dict['Title']:
              load_dict['Title'] = load_dict['Title'].replace("[\\p{P}+~$`^=|×]"," ")
            if load_dict['Comments_Embedding']:
              load_dict['Comments_Embedding'] = load_dict['Comments_Embedding'].replace("[\\p{P}+~$`^=|×]"," ")
            if load_dict['Body']:
              load_dict['Body'] = load_dict['Body'].replace("[\\p{P}+~$`^=|<×]"," ")
            if load_dict['Review_Comments_Embedding']:
              load_dict['Review_Comments_Embedding'] = load_dict['Review_Comments_Embedding'].replace("[\\p{P}+~$`^=|×]"," ")
            pattern = re.compile('^[a-zA-Z0-9]+$')
            size_TAB = 0
            size_CAR = 0
            list_Title=[0,0,0,0,0,0,0,0,0,0]
            if load_dict['Title']:
              for item in load_dict['Title'].split(" "):
                if pattern.match(item) and item in model:
                  list_Title = [a+b for a,b in zip(model[item],list_Title)]
            if load_dict['Title']:
              size_TAB = size_TAB + len(load_dict['Title'].split(" "))
            list_Comments_Embedding = [0,0,0,0,0,0,0,0,0,0]
            if load_dict['Comments_Embedding']:
              for item in load_dict['Comments_Embedding'].split(" "):
                if pattern.match(item) and item in model:
                  list_Comments_Embedding = [a+b for a,b in zip(model[item],list_Comments_Embedding)]
            if load_dict['Comments_Embedding']:
              size_CAR = size_CAR + len(load_dict['Comments_Embedding'].split(" "))
            list_Body=[0,0,0,0,0,0,0,0,0,0]
            if load_dict['Body']:
              for item in load_dict['Body'].split(" "):
                if pattern.match(item) and item in model:
                  list_Body = [a+b for a,b in zip(model[item],list_Body)]
            if load_dict['Body']:        
              size_TAB = size_TAB + len(load_dict['Body'].split(" "))
            list_Review_Comments_Embedding=[0,0,0,0,0,0,0,0,0,0]
            if load_dict['Review_Comments_Embedding']:
              for item in load_dict['Review_Comments_Embedding'].split(" "):
                if pattern.match(item) and item in model:
                  list_Review_Comments_Embedding = [a+b for a,b in zip(model[item],list_Review_Comments_Embedding)]
            if load_dict['Review_Comments_Embedding']:    
              size_CAR = size_CAR + len(load_dict['Review_Comments_Embedding'].split(" "))
            list_TAB = [a+b for a,b in zip(list_Title,list_Body)]
            for value in list_TAB:
              if value != 0:
                value = value/size_TAB
            load_dict['X1_0'] = list_TAB[0]
            load_dict['X1_1'] = list_TAB[1]
            load_dict['X1_2'] = list_TAB[2]
            load_dict['X1_3'] = list_TAB[3]
            load_dict['X1_4'] = list_TAB[4]
            load_dict['X1_5'] = list_TAB[5]
            load_dict['X1_6'] = list_TAB[6]
            load_dict['X1_7'] = list_TAB[7]
            load_dict['X1_8'] = list_TAB[8]
            load_dict['X1_9'] = list_TAB[9]
            list_CAR = [a+b for a,b in zip(list_Comments_Embedding,list_Review_Comments_Embedding)]
            for value in list_CAR:
              if value != 0:
                value = value/size_CAR
            load_dict['X2_0'] = list_CAR[0]
            load_dict['X2_1'] = list_CAR[1]
            load_dict['X2_2'] = list_CAR[2]
            load_dict['X2_3'] = list_CAR[3]
            load_dict['X2_4'] = list_CAR[4]
            load_dict['X2_5'] = list_CAR[5]
            load_dict['X2_6'] = list_CAR[6]
            load_dict['X2_7'] = list_CAR[7]
            load_dict['X2_8'] = list_CAR[8]
            load_dict['X2_9'] = list_CAR[9]
            f_csv.writerow(load_dict)
        except Exception as e:
            print(e)
            continue    



def generate_response_data(project_full_name, start_time, end_time):
  model = gensim.models.Word2Vec.load('word2vec_model')
  count = 0
  file = open(raw_data,"r")
  fieldnames = []
  for line in file.readlines():
    load_dict = json.loads(line)
    fieldnames = list(load_dict.keys())
    break
  fieldnames.append('X1_0')
  fieldnames.append('X1_1')
  fieldnames.append('X1_2')
  fieldnames.append('X1_3')
  fieldnames.append('X1_4')
  fieldnames.append('X1_5')
  fieldnames.append('X1_6')
  fieldnames.append('X1_7')
  fieldnames.append('X1_8')
  fieldnames.append('X1_9')
  fieldnames.append('X2_0')
  fieldnames.append('X2_1')
  fieldnames.append('X2_2')
  fieldnames.append('X2_3')
  fieldnames.append('X2_4')
  fieldnames.append('X2_5')
  fieldnames.append('X2_6')
  fieldnames.append('X2_7')
  fieldnames.append('X2_8')
  fieldnames.append('X2_9')
  fieldnames.append('wait_time_up')
  fieldnames.append('wait_time_label')
  file = open(raw_data,"r")
  with open(response_data,'w',newline='',errors='ignore') as f:
    f_csv = csv.DictWriter(f,fieldnames=fieldnames)
    f_csv.writeheader()
    for line in file.readlines():
      try:
          load_dict = json.loads(line)
          #print(load_dict['url'])
          if load_dict['url'].find(project_full_name) != -1 and time.strptime(str(load_dict['Timeline'][1]['Created_At'])[2:-2],"%Y-%m-%dT%H:%M:%SZ") >= start_time:
              continue
          Wait_Time = {}
          #print(count)
          time_length = int((time.mktime(time.strptime(str(load_dict['Timeline'][1]['Created_At'])[2:-2],"%Y-%m-%dT%H:%M:%SZ"))-time.mktime(time.strptime(str(load_dict['Timeline'][0]['Created_At'])[2:-2],"%Y-%m-%dT%H:%M:%SZ")))/3600/24)
          #print(len(load_dict['Timeline']))
          for item in range(1,len(load_dict['Timeline'])):
            #print(item)
            tmp_list = eval(str(load_dict['Timeline'][item]['Created_At']))
            #print(tmp_list)
            for i in tmp_list:
              Wait_Time[int((time.mktime(time.strptime(i,"%Y-%m-%dT%H:%M:%SZ"))-time.mktime(time.strptime(str(load_dict['Timeline'][0]['Created_At'])[2:-2],"%Y-%m-%dT%H:%M:%SZ")))/3600/24)] = True
          if load_dict['Title']:
            load_dict['Title'] = load_dict['Title'].replace("[\\p{P}+~$`^=|×]"," ")
          if load_dict['Comments_Embedding']:
            load_dict['Comments_Embedding'] = load_dict['Comments_Embedding'].replace("[\\p{P}+~$`^=|×]"," ")
          if load_dict['Body']:
            load_dict['Body'] = load_dict['Body'].replace("[\\p{P}+~$`^=|<×]"," ")
          if load_dict['Review_Comments_Embedding']:
            load_dict['Review_Comments_Embedding'] = load_dict['Review_Comments_Embedding'].replace("[\\p{P}+~$`^=|×]"," ")
          pattern = re.compile('^[a-zA-Z0-9]+$')
          size_TAB = 0
          size_CAR = 0
          list_Title=[0,0,0,0,0,0,0,0,0,0]
          if load_dict['Title']:
            for item in load_dict['Title'].split(" "):
              if pattern.match(item) and item in model:
                list_Title = [a+b for a,b in zip(model[item],list_Title)]
          if load_dict['Title']:
            size_TAB = size_TAB + len(load_dict['Title'].split(" "))
          list_Comments_Embedding = [0,0,0,0,0,0,0,0,0,0]
          if load_dict['Comments_Embedding']:
            for item in load_dict['Comments_Embedding'].split(" "):
              if pattern.match(item) and item in model:
                list_Comments_Embedding = [a+b for a,b in zip(model[item],list_Comments_Embedding)]
          if load_dict['Comments_Embedding']:
            size_CAR = size_CAR + len(load_dict['Comments_Embedding'].split(" "))
          list_Body=[0,0,0,0,0,0,0,0,0,0]
          if load_dict['Body']:
            for item in load_dict['Body'].split(" "):
              if pattern.match(item) and item in model:
                list_Body = [a+b for a,b in zip(model[item],list_Body)]
          if load_dict['Body']:        
            size_TAB = size_TAB + len(load_dict['Body'].split(" "))
          list_Review_Comments_Embedding=[0,0,0,0,0,0,0,0,0,0]
          if load_dict['Review_Comments_Embedding']:
            for item in load_dict['Review_Comments_Embedding'].split(" "):
              if pattern.match(item) and item in model:
                list_Review_Comments_Embedding = [a+b for a,b in zip(model[item],list_Review_Comments_Embedding)]
          if load_dict['Review_Comments_Embedding']:    
            size_CAR = size_CAR + len(load_dict['Review_Comments_Embedding'].split(" "))
          list_TAB = [a+b for a,b in zip(list_Title,list_Body)]
          for value in list_TAB:
            if value != 0:
              value = value/size_TAB
          load_dict['X1_0'] = list_TAB[0]
          load_dict['X1_1'] = list_TAB[1]
          load_dict['X1_2'] = list_TAB[2]
          load_dict['X1_3'] = list_TAB[3]
          load_dict['X1_4'] = list_TAB[4]
          load_dict['X1_5'] = list_TAB[5]
          load_dict['X1_6'] = list_TAB[6]
          load_dict['X1_7'] = list_TAB[7]
          load_dict['X1_8'] = list_TAB[8]
          load_dict['X1_9'] = list_TAB[9]
          list_CAR = [a+b for a,b in zip(list_Comments_Embedding,list_Review_Comments_Embedding)]
          for value in list_CAR:
            if value != 0:
              value = value/size_CAR
          load_dict['X2_0'] = list_CAR[0]
          load_dict['X2_1'] = list_CAR[1]
          load_dict['X2_2'] = list_CAR[2]
          load_dict['X2_3'] = list_CAR[3]
          load_dict['X2_4'] = list_CAR[4]
          load_dict['X2_5'] = list_CAR[5]
          load_dict['X2_6'] = list_CAR[6]
          load_dict['X2_7'] = list_CAR[7]
          load_dict['X2_8'] = list_CAR[8]
          load_dict['X2_9'] = list_CAR[9]
          if time_length < 10:
              time_length = time_length
          else:
              time_length = 10
          for wait_time in range(time_length):
              load_dict['wait_time_up'] = wait_time
              load_dict['wait_time_label'] = False
              load_dict['Day']=((time.strptime(str(load_dict['Timeline'][0]['Created_At'])[2:-2],"%Y-%m-%dT%H:%M:%SZ"))[6]+wait_time)%7
              if wait_time in Wait_Time.keys():
                  load_dict['wait_time_label'] = True
              f_csv.writerow(load_dict)
      except Exception as e:
          print(e)
          continue

def generate_sort_data(project_full_name, start_time, end_time):
  model = gensim.models.Word2Vec.load('word2vec_model')
  count = 0
  file = open(raw_data,"r")
  fieldnames = []
  for line in file.readlines():
    load_dict = json.loads(line)
    fieldnames = list(load_dict.keys())
    break
  fieldnames.append('X1_0')
  fieldnames.append('X1_1')
  fieldnames.append('X1_2')
  fieldnames.append('X1_3')
  fieldnames.append('X1_4')
  fieldnames.append('X1_5')
  fieldnames.append('X1_6')
  fieldnames.append('X1_7')
  fieldnames.append('X1_8')
  fieldnames.append('X1_9')
  fieldnames.append('X2_0')
  fieldnames.append('X2_1')
  fieldnames.append('X2_2')
  fieldnames.append('X2_3')
  fieldnames.append('X2_4')
  fieldnames.append('X2_5')
  fieldnames.append('X2_6')
  fieldnames.append('X2_7')
  fieldnames.append('X2_8')
  fieldnames.append('X2_9')
  fieldnames.append('wait_time_up')
  fieldnames.append('Response_Label')
  os.mkdir('paixu/data/{}_{}_{}'.format(project_full_name.replace('/','_'), start_time[0],start_time[1]))
  for i in range(0,32):
      date_time_start=time.mktime(start_time)+3600*24*i
      date_time_start = time.localtime(date_time_start)
      date_time_start=time.strptime(time.strftime('%Y-%m-%dT%H:%M:%SZ',date_time_start),'%Y-%m-%dT%H:%M:%SZ')
      if date_time_start == end_time: break
      date_time_end = time.mktime(start_time)+3600*24*(i+1)
      date_time_end = time.localtime(date_time_end)
      date_time_end=time.strptime(time.strftime('%Y-%m-%dT%H:%M:%SZ', date_time_end),'%Y-%m-%dT%H:%M:%SZ')
      file = open(raw_data,"r")
      with open('paixu/data/{}_{}_{}/{}_{}_{}.csv'.format(project_full_name.replace('/','_'), start_time[0],start_time[1], date_time_start[0],date_time_start[1],date_time_start[2]),'w',newline='',errors='ignore') as f:
          f_csv = csv.DictWriter(f,fieldnames=fieldnames)
          f_csv.writeheader()
          for line in file.readlines():
            try:
              load_dict = json.loads(line)
              if load_dict['url'].find(project_full_name) == -1:
                   continue
              if time.strptime(str(load_dict['Timeline'][1]['Created_At'])[2:-2],"%Y-%m-%dT%H:%M:%SZ") >date_time_start and time.strptime(str(load_dict['Timeline'][0]['Created_At'])[2:-2],"%Y-%m-%dT%H:%M:%SZ") < date_time_start:
              #    print('no1')
                  load_dict['Response_Label'] = 0
                  load_dict['wait_time_up'] = int((time.mktime(date_time_start)-time.mktime(time.strptime(str(load_dict['Timeline'][0]['Created_At'])[2:-2],"%Y-%m-%dT%H:%M:%SZ")))/3600/24)
                  load_dict['Wait_Time'] = (time.mktime(date_time_start)-time.mktime(time.strptime(str(load_dict['Timeline'][0]['Created_At'])[2:-2],"%Y-%m-%dT%H:%M:%SZ")))/3600/24
                  load_dict['Day'] = date_time_start[6]
                  for k in range(1,len(load_dict['Timeline'])):
                      tmp_list = eval(str(load_dict['Timeline'][k]['Created_At']))
                      for j in tmp_list:
                          if time.strptime(j,"%Y-%m-%dT%H:%M:%SZ")>=date_time_start and time.strptime(j,"%Y-%m-%dT%H:%M:%SZ")<= date_time_end:
                             load_dict['Response_Label'] = 1
                  if load_dict['Title']:
                    load_dict['Title'] = load_dict['Title'].replace("[\\p{P}+~$`^=|×]"," ")
                  if load_dict['Comments_Embedding']:
                    load_dict['Comments_Embedding'] = load_dict['Comments_Embedding'].replace("[\\p{P}+~$`^=|×]"," ")
                  if load_dict['Body']:
                    load_dict['Body'] = load_dict['Body'].replace("[\\p{P}+~$`^=|<×]"," ")
                  if load_dict['Review_Comments_Embedding']:
                    load_dict['Review_Comments_Embedding'] = load_dict['Review_Comments_Embedding'].replace("[\\p{P}+~$`^=|×]"," ")
                  pattern = re.compile('^[a-zA-Z0-9]+$')
                  size_TAB = 0
                  size_CAR = 0
                  list_Title=[0,0,0,0,0,0,0,0,0,0]
                  if load_dict['Title']:
                    for item in load_dict['Title'].split(" "):
                      if pattern.match(item) and item in model:
                        list_Title = [a+b for a,b in zip(model[item],list_Title)]
                  if load_dict['Title']:
                    size_TAB = size_TAB + len(load_dict['Title'].split(" "))
                  list_Comments_Embedding = [0,0,0,0,0,0,0,0,0,0]
                  if load_dict['Comments_Embedding']:
                    for item in load_dict['Comments_Embedding'].split(" "):
                      if pattern.match(item) and item in model:
                        list_Comments_Embedding = [a+b for a,b in zip(model[item],list_Comments_Embedding)]
                  if load_dict['Comments_Embedding']:
                    size_CAR = size_CAR + len(load_dict['Comments_Embedding'].split(" "))
                  list_Body=[0,0,0,0,0,0,0,0,0,0]
                  if load_dict['Body']:
                    for item in load_dict['Body'].split(" "):
                      if pattern.match(item) and item in model:
                        list_Body = [a+b for a,b in zip(model[item],list_Body)]
                  if load_dict['Body']:        
                    size_TAB = size_TAB + len(load_dict['Body'].split(" "))
                  list_Review_Comments_Embedding=[0,0,0,0,0,0,0,0,0,0]
                  if load_dict['Review_Comments_Embedding']:
                    for item in load_dict['Review_Comments_Embedding'].split(" "):
                      if pattern.match(item) and item in model:
                        list_Review_Comments_Embedding = [a+b for a,b in zip(model[item],list_Review_Comments_Embedding)]
                  if load_dict['Review_Comments_Embedding']:    
                    size_CAR = size_CAR + len(load_dict['Review_Comments_Embedding'].split(" "))
                  list_TAB = [a+b for a,b in zip(list_Title,list_Body)]
                  for value in list_TAB:
                    if value != 0:
                      value = value/size_TAB
                  load_dict['X1_0'] = list_TAB[0]
                  load_dict['X1_1'] = list_TAB[1]
                  load_dict['X1_2'] = list_TAB[2]
                  load_dict['X1_3'] = list_TAB[3]
                  load_dict['X1_4'] = list_TAB[4]
                  load_dict['X1_5'] = list_TAB[5]
                  load_dict['X1_6'] = list_TAB[6]
                  load_dict['X1_7'] = list_TAB[7]
                  load_dict['X1_8'] = list_TAB[8]
                  load_dict['X1_9'] = list_TAB[9]
                  list_CAR = [a+b for a,b in zip(list_Comments_Embedding,list_Review_Comments_Embedding)]
                  for value in list_CAR:
                    if value != 0:
                      value = value/size_CAR
                  load_dict['X2_0'] = list_CAR[0]
                  load_dict['X2_1'] = list_CAR[1]
                  load_dict['X2_2'] = list_CAR[2]
                  load_dict['X2_3'] = list_CAR[3]
                  load_dict['X2_4'] = list_CAR[4]
                  load_dict['X2_5'] = list_CAR[5]
                  load_dict['X2_6'] = list_CAR[6]
                  load_dict['X2_7'] = list_CAR[7]
                  load_dict['X2_8'] = list_CAR[8]
                  load_dict['X2_9'] = list_CAR[9]
                  f_csv.writerow(load_dict)
            except Exception as e:
              print(e)
              continue

def train_accept_model():
  def read_data():
    data_path = accept_data
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
    data_label = data['Label']
    train_data,test_data,train_label,test_label = train_test_split(data,data_label,test_size=0.1,stratify=data_label)
    #print(len(train_data))
    #print(len(test_data))
    train_data = data
    train_label = data_label
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
     'Wait_Time',  'Comments_Count', 'Comments_Per_Closed_PR', 'Participants_Count', 'Accept_Num',
     'Following', 'Wednesday', 'Project_Age', 'Prev_PRs', 'Intra_Branch', 'Files_Changed',  'Closed_Num',
      'Additions_Per_Week', 'Label_Count', 'Open_Issues', 'Organization_Core_Member', 
    'Commits_PR', 'Point_To_IssueOrPR', 'Last_Comment_Mention', 'Stars', 'Close_Latency']
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
    'XGBoost':'XGBoost',
    }
  def XGBoost():
    try:
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
      model.save_model('xgb.model')
      return
    except:
       print('die')
    
  def trained_and_predict_location(clf, X_train, y_train, X_test, y_test):
    if clf == "XGBoost":
      try:
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
        model.save_model('xgb.model')
        return 
      except Exception as e:
        print(e)
        print('die')
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
        train_data,train_label,test_data,test_label = split()
        X_train,y_train,X_test,y_test = train_and_test_to_vec(train_data,train_label,test_data,test_label)
        trained_and_predict_location(clf, X_train, y_train, X_test, y_test)


  traind_and_predict()

def train_response_model():
  def read_data():
    data_path = response_data
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
    train_data = data
    train_label = data_label
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
    'XGBoost':'XGBoost',
    }
  def XGBoost():
    try:
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
      xgb.save_model('xgb_time.model')
      return 
    except:
       print('die')
    
  def trained_and_predict_location(clf, X_train, y_train, X_test, y_test):
    if clf == "XGBoost":
      try:
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
        model.save_model('xgb_time.model')
        return 
      except Exception as e:
         print(e)
         print('die')
    else:
      print('fitting....')
      clf = clf.fit(X_train, y_train)
      print('pridict....')
      return clf.predict(X_test)

  def traind_and_predict():
    classifiers = get_classifiers()
    for name,clf in classifiers.items():
        train_data,train_label,test_data,test_label = split()
        X_train,y_train,X_test,y_test = train_and_test_to_vec(train_data,train_label,test_data,test_label)
        trained_and_predict_location(clf, X_train, y_train, X_test, y_test)
        #score = accuracy_score(y_test, predicted)
        #fpr, tpr, thresholds = metrics.roc_curve(y_test, predicted, pos_label=2)
        #score = auc(fpr, tpr)
  traind_and_predict()

def predict_and_sort(project_full_name, start_time, end_time):
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
    os.mkdir('paixu/result/result_{}_{}_{}'.format(project_full_name.replace('/','_'), start_time[0], start_time[1]))
    classifiers = get_classifiers()
    for name,clf in classifiers.items():
      for i in range(0,32):
          try:
            date_time_start=time.mktime(start_time)+3600*24*i
            date_time_start = time.localtime(date_time_start)
            date_time_start=time.strptime(time.strftime('%Y-%m-%dT%H:%M:%SZ',date_time_start),'%Y-%m-%dT%H:%M:%SZ')
            if date_time_start == end_time: break
            name_file="paixu/data/{}_{}_{}/{}_{}_{}.csv".format(project_full_name.replace('/','_'), date_time_start[0], date_time_start[1], date_time_start[0], date_time_start[1], date_time_start[2])
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
            result.to_csv('paixu/result/result_{}_{}_{}/{}_{}_{}.csv'.format(project_full_name.replace('/','_'), start_time[0], start_time[1], date_time_start[0], date_time_start[1], date_time_start[2]))
        except Exception as e:
            print(e)
            continue
  traind_and_predict()


if __name__ == '__main__':
  # "%Y-%m-%dT%H:%M:%SZ"
  date_list=[
              "2017-09-01T00:00:00Z",
              "2017-10-01T00:00:00Z",
              "2017-11-01T00:00:00Z",
              "2017-12-01T00:00:00Z",
              "2018-01-01T00:00:00Z",
              "2018-02-01T00:00:00Z",
              "2018-03-01T00:00:00Z"
            ]
  project_list=[
                'NixOS/nixpkgs', 
                'django/django',
                'facebook/react',
                'angular/angular.js',
                'saltstack/salt',
                'cms-sw/cmssw',
                'laravel/framework',
                'scikit-learn/scikit-learn',
                'cdnjs/cdnjs',
                'hashicorp/terraform',
                'yiisoft/yii2',
                'githubschool/open-enrollment-classes-introduction-to-github',
                'kubernetes/kubernetes',
                'rust-lang/rust',
                'rails/rails',
                'moby/moby',
                'symfony/symfony',
                'TheOdinProject/curriculum',
                'opencv/opencv',
                'tensorflow/tensorflow',
                'pandas-dev/pandas'
                ]
  for i in range(0, len(date_list)-1):
    for j in project_list:
      print('{}_{} processing...'.format(j,date_list[i]))
      start_time=time.strptime(date_list[i],"%Y-%m-%dT%H:%M:%SZ")
      end_time=time.strptime(date_list[i+1],"%Y-%m-%dT%H:%M:%SZ")

      print('{}_{} generate_accept_data...'.format(j,date_list[i]))
      generate_accept_data(j, start_time, end_time)

      print('{}_{} generate_response_data...'.format(j,date_list[i]))
      generate_response_data(j, start_time, end_time)

      print('{}_{} generate_sort_data...'.format(j,date_list[i]))
      generate_sort_data(j, start_time, end_time)

      print('{}_{} train_accept_model...'.format(j,date_list[i]))
      train_accept_model()

      print('{}_{} train_response_model...'.format(j,date_list[i]))
      train_response_model()

      print('{}_{} predict_and_sort...'.format(j,date_list[i]))
      predict_and_sort(j, start_time, end_time)
