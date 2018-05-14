#coding=utf-8  
import time
import json
import re
import csv
import datetime
import matplotlib as mpl 
mpl.use('Agg')
import matplotlib.pyplot as plt
time_list=[]
open_list=[]
accept_list=[]
response_list=[]

for i in range(1,32):
    print(i)
    open_tmp=0
    accept_tmp=0
    response_tmp=0
    date_start_string=''
    date_end_string=''
    if i< 10:
        date_start_string = '2017-12-0' + str(i)
    else:
        date_start_string = '2017-12-' + str(i)
    if i + 1 <10:
        date_end_string='2017-12-0' + str(i+1)
    elif i+1<=31:
        date_end_string='2017-12-' + str(i+1)
    elif i+1==32:
        date_end_string='2018-01-01'
    date_time_start = time.strptime("{}T00:00:00Z".format(date_start_string),"%Y-%m-%dT%H:%M:%SZ")
    date_time_end = time.strptime("{}T00:00:00Z".format(date_end_string),"%Y-%m-%dT%H:%M:%SZ")
    file = open("filter-raw-data.txt","r")
    for line in file.readlines():
        try:
            loop_flag=False
            load_dict = json.loads(line)
            pr_start_time=time.strptime(str(load_dict['Timeline'][0]['Created_At'])[2:-2],"%Y-%m-%dT%H:%M:%SZ")
            pr_end_time=time.strptime(str(load_dict['Timeline'][1]['Created_At'])[2:-2],"%Y-%m-%dT%H:%M:%SZ")
            if pr_start_time<=date_time_end and pr_end_time>= date_time_start:
                open_tmp +=1
            else: continue
                #if int(load_dict['Label']) == 1:
                #    accept_tmp+=1
            if pr_end_time<=date_time_end and pr_end_time>=date_time_start:
                if int(load_dict['Label'])==1:
                    loop_flag=True
                    accept_tmp+=1
                response_tmp+=1
                #continue
            #loop_flag=False
            for j in range(2,len(load_dict['Timeline'])):
                timeline = eval(str(load_dict['Timeline'][j]['Created_At']))
                #if loop_flag: 
                #    break
                for k in timeline:
                    time_tmp = time.strptime(k,"%Y-%m-%dT%H:%M:%SZ")
                    if time_tmp<=date_time_end and time_tmp>=date_time_start:
                        if int(load_dict['Label']) ==1 and not loop_flag:
                            accept_tmp+=1
                        response_tmp+=1
        except Exception as e:
            print(e)
    open_list.append(open_tmp)
    accept_list.append(accept_tmp)
    response_list.append(response_tmp)
    time_list.append(i)
print(open_list)
print(accept_list)
print(time_list)
plt.plot(time_list,open_list,marker='o',markerfacecolor='blue',label = 'open number')
plt.plot(time_list,accept_list,marker='*',markerfacecolor='green',label = 'accpet number')
plt.plot(time_list,response_list,marker='^',markerfacecolor='red',label = 'response number')
plt.legend()
plt.xlabel('time')
plt.ylabel('number')
plt.savefig("filename.png")
