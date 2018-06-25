import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def Get_Average(list):
	sum = 0
	for item in list:
		sum += item
	return sum/len(list)
result={}
with open('result.txt','r') as f:
    for line in f.readlines():
        tmp=json.loads(line.strip())
        print(tmp['name'])
        if tmp['project'] in result.keys():
            if Get_Average(result[tmp['project']]['score']) < Get_Average(tmp['score']):
                result[tmp['project']] = tmp
        else:
            result[tmp['project']]=tmp
all_data=[]
all_name=[]
tmp=[]
for k in result.keys():
    tmp.append({'data':result[k]['score'], 'name':result[k]['name']})
tmp.sort(key=lambda x:Get_Average(x['data']), reverse=True)
for i in tmp:
    print(Get_Average(i['data']))
for i in range(1,11):
    all_data.append(tmp[i]['data'])
    all_name.append(tmp[i]['name'])
    #print(json.dumps({'name':tmp[i]['name'], 'data':tmp[i]['data']}))
    print(len(tmp[i]['data']))
# for k in result.keys():
#     all_data.append(result[k]['score'])
#     if result[k]['project'] =='open-enrollment-classes-introduction-to-github':
#         all_name.append(result[k]['name'].replace('open-enrollment-classes-introduction-to-github','open-enrollment-classes-\nintroduction-to-github'))
#     else:
#         all_name.append(result[k]['name'])
fig = plt.figure(figsize=(15,15))
plt.xticks(rotation=45)
plt.boxplot(all_data,
            notch=False, # box instead of notch shape
            sym='rs',    # red squares for outliers
            vert=True)   # vertical box aligmnent
plt.xticks([y+1 for y in range(len(all_data))], all_name, fontsize=13)
plt.xlabel('Project-Month')
t = plt.title('AP')
#plt.show()
plt.savefig('result.png')
