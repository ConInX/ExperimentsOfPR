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
file_organization = open('../data/organization.txt', 'r')
file_project = open('../data/project.txt', 'a+')
file_log = open('log_url.txt','a+')
logging.getLogger().setLevel(logging.INFO)
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
			'2719445c05991f1e5febfef1a09ca8befd042ddd'
		]
def generateProjects(file, token, identification):
	# Read File
	global lines
	name = identification
	while True:
		if mutex.acquire(True):
			line = file.readline()
			lines +=1
			logging.info('Lines: {}, From: {}'.format(lines, name))
			mutex.release()
		if line == '': 
			logging.info(' ------------------------  done. From: {} ---------------------------'.format(name))
			break
		organization = json.loads(line.strip())
		loop_url = organization['repos_url']+'?&access_token='+token
		while True:
			try:
				file_log.write('From: {}, loop_url: {} \n'.format(name, loop_url))
				response = requests.get(loop_url)
				if response.status_code == requests.codes.ok:
					data = response.json()
					for tmp in data:
						tmp_data = {}
						tmp_data['id'] = tmp['id']
						tmp_data['full_name'] = tmp['full_name']
						tmp_data['name'] = tmp['name']
						tmp_data['pulls_url'] = tmp['pulls_url']
						tmp_data['forks_count'] = tmp['forks_count']
						tmp_data['size'] = tmp['size']
						file_project.write(json.dumps(tmp_data)+'\n')
					if 'next' in response.links.keys(): 
						loop_url = response.links['next']['url']
					else:
						break
					if int(response.headers['X-RateLimit-Remaining']) == 0:
						logging.warning('Sleep: {}s, From: {}.'.format(600, name))
						time.sleep(600)
				elif str(response.status_code) == '404':
					break
				else:
					logging.warning('Status: {}, From: {}, Sleep: {}s, '.format(response.status_code, name, 600))
					time.sleep(600)
			except Exception as e:
				logging.warning('$$$$$$$$ From: {}, Exception: {}, Sleep {}s'.format(name, e, 60))
				time.sleep(60)


for i in range(len(token_list)):
	t = threading.Thread(target = generateProjects, args=(file_organization, token_list[i], i,))
	t.start()


logging.info('-------------------------finished---------------------------')