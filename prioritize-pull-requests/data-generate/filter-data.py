# -*- coding:utf-8 -*-
import itertools
import json
import logging
import requests
import time
import threading
#创建锁
mutex = threading.Lock()
lines=0
#锁定
#mutex.acquire([timeout])
#释放
#mutex.release()
input_file = open('raw-data/raw-data.txt','r')
error_log = open('error_log.txt','w')
filter_data = open('raw-data/filter-raw-data.txt','w')
word2vec_data = open('../word2vec/train-data.txt','w')
filter_project=['apache/spark','nodejs/node','udacity/create-your-own-adventure','facebook/react-native']

token_list=[
                    'be77d477afb39c6345e649d0288b5c83ee024c4f',
                    'd0ee97a857d1e8e4c4918593bacfd69eaf011612',
                    '4cd06100b2c2234a3f02151e4f1ee1570c5b8a19',
                    'd4b89ddd287e96b9a68774502ab6809521229db7',
                    'e21995fb5f942669b2a3edd2b3b754e568e33217',
                    '3a867c56d662d7bd9778d35b653a7b19790b6777',
                    'fe5583b93cbcd0fc02d751530cfc5d489804524b',
                    'e5bb25ef369589e90e200cb1a528f1820bdb9994',
                    '6cef8e2c458c84b776d82c3e0b29bde46fb6d2cf',
                    '2719445c05991f1e5febfef1a09ca8befd042ddd',
                    'dcefdee459ea41ffd80b0372f056ca1a7aec49a1',
                    '029e57649d47dda6f0d81d6643518ba8af153626',
                    'c19f2995cdbe35ffe335d9e94573c8b75cb9bce1'
            ]
token_iter = itertools.cycle(token_list)

def flat(tree):
    res = []
    for i in tree:
        if isinstance(i, list):
            res.extend(flat(i))
        else:
            res.append(i)
    return res

def get_request_body(url,get_headers = None):
    body=[]
    loop_url = url+'&access_token='+next(token_iter)
    #print(url)
    headers = {'Accept':'application/vnd.github.v3.text+json'} if get_headers is None else get_headers
    #print(headers)
    while True:
                try:
                    response = requests.get(loop_url, timeout=10,headers=headers)
                    if response.status_code == requests.codes.ok:
                        data = response.json()
                        if isinstance(data,dict) and 'items' in data.keys():
                            body.append(data['items'])
                        elif isinstance(data,dict) and 'errors' in data.keys():
                            error_log.write('Arguments Error:'+loop_url+'\n')
                            error_log.flush()
                        else:
                            body.append(data)
                        if 'next' in response.links.keys():
                            loop_url = response.links['next']['url']
                        else:
                            break
                        if int(response.headers['X-RateLimit-Remaining']) == 0:
                            logging.warning('Sleep: {}s because of rateLimit'.format(600))
                            time.sleep(600)
                    elif str(response.status_code) == '404' or str(response.status_code)=='451':
                        error_log.write('Status Error:'+loop_url+'\n')
                        error_log.flush()
                        break
                    elif str(response.status_code) =='403':
                        logging.warning('Status: {}, Sleep: {}s '.format(response.status_code, 60))
                        error_log.write('Status Error:'+loop_url+'\n')
                        error_log.flush()
                        time.sleep(60)
                    else:
                        logging.warning('Status：{}, Sleep: {}s'.format(response.status_code, 60))
                        error_log.write('Status Error:'+loop_url+'\n')
                        error_log.flush()
                        time.sleep(60)
                except Exception as e:
                    logging.warning('$$$$$$$$ Exception: {}, Sleep: {}s'.format(e, 60))
                    time.sleep(60)
    return flat(body)

def response_features(example):
    body = get_request_body(example['url']+'?')
    pull_request=body[0]
    timeline = []
    timeline.append({'Created_At':[pull_request['created_at']],'Type':'created'})
    timeline.append({'Created_At':[pull_request['closed_at']],'Type':'closed'})
    tmp = get_request_body('https://api.github.com/repos/{}/issues/{}/timeline?per_page=100'.format(pull_request['base']['repo']['full_name'], pull_request['number']),get_headers={'Accept':'application/vnd.github.mockingbird-preview'})
    for i in tmp:
        tmp_dict = {}
        tmp_dict['Created_At'] = []
        if i['event'] == 'labeled':
            tmp_dict['Type'] = 'labeled'
            tmp_dict['Created_At'].append(i['created_at'])
        elif i['event'] == 'unlabeled':
            tmp_dict['Type'] = 'unlabeled'
            tmp_dict['Created_At'].append(i['created_at'])
        elif 'comments' in i.keys():
            for j in i['comments']:
                tmp_dict['Type'] = 'line-commented'
                tmp_dict['Created_At'].append(j['created_at'])

        elif i['event'] == 'milestoned':
            tmp_dict['Type'] = 'milestoned'
            tmp_dict['Created_At'].append(i['created_at'])
        elif i['event'] == 'assigned':
            tmp_dict['Type'] = 'assigned'
            tmp_dict['Created_At'].append(i['created_at'])

        elif i['event'] == 'locked':
            tmp_dict['Type'] = 'locked'
            tmp_dict['Created_At'].append(i['created_at'])

        elif i['event'] =='marked_as_duplicate':
            tmp_dict['Type'] = 'marked_as_duplicate'
            tmp_dict['Created_At'].append(i['created_at'])

        elif i['event'] =='review_requested':
            tmp_dict['Type'] = 'review_requested'
            tmp_dict['Created_At'].append(i['created_at'])

        elif i['event'] =='commented':
            if i['author_association'] == 'MEMBER' or i['author_association'] == 'OWNER' or i['author_association'] == 'COLLABORATOR':
                tmp_dict['Type'] = 'commented'
                tmp_dict['Created_At'].append(i['created_at'])
            else: 
                continue
        elif i['event'] == 'merged' or i['event'] == 'closed':
            tmp_dict['Type'] = 'closed'
            tmp_dict['Created_At'].append(i['created_at'])
        else:
            continue
        timeline.append(tmp_dict)
    example['Timeline'] = timeline
    comments = get_request_body(pull_request['comments_url']+'?per_page=100')
    Fix_Bug = 0
    Point_To_IssueOrPR = 0
    if pull_request['title'] is not None and pull_request['title'].lower().find('fix') != -1 or pull_request['body_text'] is not None and pull_request['body_text'].lower().find('fix') != -1:
        Fix_Bug = 1
    if pull_request['title'] is not None and pull_request['title'].lower().find('#') != -1 or pull_request['body_text'] is not None and pull_request['body_text'].lower().find('#') != -1:
        Point_To_IssueOrPR = 1
    if comments is not None and len(comments) != 0: 
        example['Last_Comment_Mention'] = 1 if comments[-1]['body_text'].find('@') != -1 else 0
    else:
        example['Last_Comment_Mention'] = 0
    example['Contain_Fix_Bug'] = Fix_Bug
    example['Point_To_IssueOrPR'] = Point_To_IssueOrPR
    return example


logging.getLogger().setLevel(logging.INFO)

def cleanData(file, filter_data_thread, word2vec_data_thread, identification,filter_project_thread):
    # Read File
    global lines
    name = 'Thread-' + str(identification)
    while True:
        if mutex.acquire(True):
            line = file.readline()
            lines +=1
            logging.info('Example: {}, From: {}'.format(lines, name))
            mutex.release()
        if line == '': 
            logging.info(' ------------------------  done. From: {} ---------------------------'.format(name))
            break
        example = json.loads(line.strip())
        # if example['Comments_Embedding'] != '':
        #     word2vec_data_thread.write(example['Comments_Embedding']+'\n') 
        # if example['Review_Comments_Embedding'] !='':
        #     word2vec_data_thread.write(example['Review_Comments_Embedding']+'\n')
        # if example['Title'] is not None and example['Title'] != '':
        #     word2vec_data_thread.write(example['Title']+'\n')
        # if example['Body'] is not None and example['Body'] !='':
        #     word2vec_data_thread.write(example['Body']+'\n')
        flag = False
        for i in filter_project_thread:
            if example['url'].lower().find(i) != -1:
                flag=True
                break
        if not flag:
            example = response_features(example)
            filter_data_thread.write(json.dumps(example)+'\n')

for i in range(len(token_list)):
    t = threading.Thread(target = cleanData, args=(input_file, filter_data, word2vec_data, i,filter_project))
    t.start()

logging.info('-------------------------finished---------------------------')
