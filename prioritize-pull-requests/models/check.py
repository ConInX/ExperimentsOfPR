import csv
import numpy as np
import pandas as pd
import time
data = pd.read_csv('accept_without_k.csv')
url = data['url']
Timeline = data['Timeline']
for i in range(len(url)):
	if url[i].find('kubernetes/kubernetes')!=-1:
		tmp = eval(Timeline[i])
		if time.strptime(str(tmp[1]['Created_At'])[2:-2],"%Y-%m-%dT%H:%M:%SZ") >= time.strptime("2018-01-01T00:00:00Z","%Y-%m-%dT%H:%M:%SZ"):
			print('yes')
