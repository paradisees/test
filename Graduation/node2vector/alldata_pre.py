import csv
import pandas as pd
from Graduation.node2vector.n2_similarity import node_similarity
method='cos'
#method='pearson'
threshold=0.4
num=32
similarity=node_similarity(threshold,method,num)
all=[]
#label=[]
feature=[]
#print(similarity)
with open('/Users/hhy/Desktop/1/test.csv','r',encoding='utf-8_sig') as f:
    csv_reader=csv.reader(f)
    for x in csv_reader:
        all.append(list(map(float,x[:])))
        #label.append(x[-1])
with open('/Users/hhy/Desktop/1/feature.csv','r',encoding='utf-8_sig') as f:
    csv_reader=csv.reader(f)
    for x in csv_reader:
        feature.append(x[0])
fea_number={}
with open('/Users/hhy/Desktop/1/feature.csv','r',encoding='utf-8_sig') as f:
    csv_reader=csv.reader(f)
    i=0
    for x in csv_reader:
        fea_number[x[0]]=i
        i+=1
def f():
    #新建一个次数矩阵，记录每个数据上共有多少个数据相加，然后求均值
    number=[[1 for i in range(len(all[0])-1)] for j in range(len(all))]
    for i in range(len(all)):
        for j in range(len(all[0])-1):
            if all[i][j]==1.0 and feature[j] in similarity.keys():
                tmp=similarity[feature[j]]
                for key in tmp.keys():
                    if all[i][fea_number[key]]!=1.0:
                        #一般不用管abs，基本没有数据 if abs(tmp[key])>all[i][fea_number[key]]:
                        if tmp[key]>all[i][fea_number[key]]:
                            all[i][fea_number[key]]=tmp[key]
                        #all[i][fea_number[key]] += tmp[key]
                        #number[i][fea_number[key]]+=1
    for i in range(len(all)):
        for j in range(len(all[0])-1):
            all[i][j]/=number[i][j]
    return all
all=f()

data = pd.DataFrame(all)
data.to_csv('/Users/hhy/Desktop/test.csv',encoding='utf-8-sig',header=False,index=False)


