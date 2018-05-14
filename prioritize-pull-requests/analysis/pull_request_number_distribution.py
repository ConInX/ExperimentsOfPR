# -*- coding:utf-8 -*-
import json

import matplotlib
matplotlib.use('Agg')
file_input = '../data/project_pull_request_number.txt'
file_output = 'pull_request_number.txt'
dict ={}

height = []
with open(file_input, 'r') as f :
	for line in f.readlines():
		project = json.loads(line.strip())
		if int(project['pull_request_number']) != 0:
			height.append(int(project['pull_request_number']))
		if project['pull_request_number'] in dict.keys():
			dict[project['pull_request_number']] +=1
		else:
			dict[project['pull_request_number']] =1

print(dict[0])
with open(file_output, 'w') as f:
	for key in sorted(dict.keys()):
 		f.write('{}\t{}\n'.format(key,dict[key]))

#import matplotlib.pyplot as plt

#plt.hist(height, 1000, normed=True, histtype='step', cumulative=True)
#plt.show()
