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
results_accept=[]
results_response=[]
results_accept_response=[]
for i in range(1,32):
	if i < 10:
		name_file = 'result_2018-01-0' + str(i) + '.csv'
	else:
		name_file = 'result_2018-01-' + str(i) + '.csv'
	data = pd.read_csv(name_file)
	score = data['result']
	response_label=data['Response_Label']
	accept_label=data['Label']
	mAP_accept=0
	mAP_response=0
	mAP_accept_response=0
	positive_count_accept=0
	positive_count_response=0
	positive_count_accept_response=0
	for j in range(0,len(score)):
		if j >199: break
		if accept_label[j]==1:
			positive_count_accept+=1
			mAP_accept+= positive_count_accept/(j+1)
		if response_label[j]==1:
			positive_count_response+=1
			mAP_response+= positive_count_response/(j+1)
		if accept_label[j]==1 and response_label[j] == 1:
			positive_count_accept_response+=1
			mAP_accept_response+= positive_count_accept_response/(j+1)
	results_accept.append(mAP_accept/positive_count_accept if positive_count_accept !=0 else 0)
	results_response.append(mAP_response/positive_count_response if positive_count_response !=0 else 0)
	results_accept_response.append(mAP_accept_response/positive_count_accept_response if positive_count_accept_response !=0 else 0)
print('{},AVG:{},var:{} ,AVG:{}'.format(results_accept,np.mean(results_accept),np.var(results_accept),get_filter_avg(results_accept)))
print('{},AVG:{} ，var:{},AVG:{}'.format(results_response,np.mean(results_response),np.var(results_response),get_filter_avg(results_response)))
print('{},AVG:{} ，var:{},AVG:{}'.format(results_accept_response,np.mean(results_accept_response),np.var(results_accept_response),get_filter_avg(results_accept_response)))
x = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31]

plt.figure()  
#plt.plot(x,results_accept,'r',label='accept')  
#plt.plot(x,results_response,'g',label='response')  
plt.plot(x,results_accept_response,'b',label='accept&response')  
plt.xlabel("time")  
plt.ylabel("value(AP)")  
plt.title("AP")
plt.legend() 
plt.savefig('map_200.png')
