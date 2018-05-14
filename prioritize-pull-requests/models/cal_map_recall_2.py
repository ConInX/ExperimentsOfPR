import csv
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  
 
def Get_Average(list):
	sum=0
	for item in list:
		sum+=item
	return sum/len(list)
def get_filter_avg(list):
	sum=0
	count=0
	for i in list:
		if i>=0.2:
			sum+=i
			count+=1
	return sum/count
def get_map(threshold):
	recall_accept_not_response=[]
	recall_close_not_response=[]
	recall_response=[]
	for i in range(1,32):
		if i < 10:
			name_file = 'result_2018-01-0' + str(i) + '.csv'
		else:
			name_file = 'result_2018-01-' + str(i) + '.csv'
		data = pd.read_csv(name_file)
		score = data['result']
		response_label=data['Response_Label']
		accept_label=data['Label']
		total_accept_not_response=0
		total_close_not_response=0
		total_response=0
		for j in range(len(accept_label)):
			if response_label[j]==1:
				total_response+=1
			else:
				if accept_label[j]==1:
					total_accept_not_response+=1
				else:
					total_close_not_response+=1
		tmp_accept_not_response=0
		tmp_close_not_response=0
		tmp_response=0
		for j in range(len(accept_label)):
			if j >threshold: break
			if response_label[j]==1:
				tmp_response+=1
			else:
				if accept_label[j]==1:
					tmp_accept_not_response+=1
				else:
					tmp_close_not_response+=1
		recall_accept_not_response.append(tmp_accept_not_response/total_accept_not_response if total_accept_not_response!=0 else 0)
		recall_close_not_response.append(tmp_close_not_response/total_close_not_response if total_close_not_response!=0 else 0)
		recall_response.append(tmp_response/total_response if total_response!=0 else 0)
	return [np.mean(recall_accept_not_response),np.mean(recall_close_not_response),np.mean(recall_response)]
x=[]
results_accept=[]
results_close=[]
results_response=[]
for i in range(0,400):
	print(i)
	x.append(i+1)
	tmp=get_map(i+1)
	results_accept.append(tmp[0])
	results_close.append(tmp[1])
	results_response.append(tmp[2])
plt.figure()  
plt.plot(x,results_accept,'r',label='accept not response')  
plt.plot(x,results_close,'g',label='close not response')  
plt.plot(x,results_response,'b', label='response')
#plt.plot(x,results,'b')  
plt.xlabel("Top-N")  
plt.ylabel("value(Average Recall)")  
plt.title("Average-Recall-Top-N")
plt.legend() 
plt.savefig('recall_2.png')
