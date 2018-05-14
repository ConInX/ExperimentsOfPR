﻿#coding=utf-8  

import json
import re
import gensim
import csv
model = gensim.models.Word2Vec.load('word2vec_model')
count = 0
file = open("filter-raw-data.txt","r")
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
file = open("filter-raw-data.txt","r")
with open('accept.csv','w',newline='',errors='ignore') as f:
  f_csv = csv.DictWriter(f,fieldnames=fieldnames)
  f_csv.writeheader()
  for line in file.readlines():
    load_dict = json.loads(line)
    count = count + 1
    print(count)
    #if load_dict['url'].find('kubernetes/kubernetes') == -1:
      #continue
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
    
