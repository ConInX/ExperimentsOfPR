#coding=utf-8
import json
import argparse
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='paixu/result', help='the directory of result file')
    parser.add_argument('--top-n', type=int, default=10, help='the directory of result file')
    args = parser.parse_args()
    #print(args.path,args.top_n)
    return args
def get_all_project_and_date(path):
    for i in os.listdir(path):
        yield (os.path.join(path,i), ''.join(i.split('_')[2:]), i.split('_')[2])
def get_all_result_file(path):
    for dirpath, dirnames, filenames in os.walk(path):
        for file in filenames:
            if os.path.splitext(file)[1] =='.csv':
                yield os.path.join(path, file)



if __name__== '__main__':
    args=parse_args()
    result_ap=[]
    for (path, name, project) in get_all_project_and_date(args.path):
        result_tmp=[]
        #print(path,name)
        dict_tmp={}
        dict_tmp['path']=path
        dict_tmp['name']=name
        dict_tmp['project']=project
        for file in get_all_result_file(path):
            data = pd.read_csv(file)
            score = data['result']
            response_label=data['Response_Label']
            accept_label=data['Label']
            # mAP_accept=0
            # mAP_response=0
            mAP_accept_response=0
            # positive_count_accept=0
            # positive_count_response=0
            positive_count_accept_response=0
            for j in range(0,len(score)):
                    if j >=args.top_n: break
                    # if accept_label[j]==1:
                    #         positive_count_accept+=1
                    #         mAP_accept+= positive_count_accept/(j+1)
                    # if response_label[j]==1:
                    #         positive_count_response+=1
                    #         mAP_response+= positive_count_response/(j+1)
                    if accept_label[j]==1 and response_label[j] == 1:
                            positive_count_accept_response+=1
                            mAP_accept_response+= positive_count_accept_response/(j+1)
            # results_accept.append(mAP_accept/positive_count_accept if positive_count_accept !=0 else 0)
            # results_response.append(mAP_response/positive_count_response if positive_count_response !=0 else 0)
            result_tmp.append(mAP_accept_response/positive_count_accept_response if positive_count_accept_response !=0 else 0)
            #if positive_count_accept_response !=0:
            #    result_tmp.append(mAP_accept_response/positive_count_accept_response)
        dict_tmp['score']=result_tmp
        print(json.dumps(dict_tmp))
        result_ap.append((name, result_tmp))
    fig = plt.figure()
    all_data = [data for name,data in result_ap]
    plt.boxplot(all_data,
                notch=False, # box instead of notch shape
                sym='rs',    # red squares for outliers
                vert=True)   # vertical box aligmnent
     
    plt.xticks([y+1 for y in range(len(all_data))], [name for name,data in result_ap])
    plt.xlabel('ap')
    t = plt.title('ap')
    #plt.show()
    plt.savefig(args.path+'.png')
