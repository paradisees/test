import csv
import numpy as np
import pandas as pd
import random
dic,fdic={},{}
def dic_construct(dic,path,flag):
    with open(path,encoding='utf-8-sig') as csvfile:
        rows = csv.reader(csvfile)
        for row in rows:
            if flag==1:
                tmp=dict(zip([row[1]],[row[2]]))
            else:
                tmp=dict(zip([row[1]],[1]))
            if row[0] not in dic:
                dic[row[0]]=tmp
            else:
                if row[1] not in dic[row[0]]:
                    for num in tmp.keys():
                        dic[row[0]][num]=tmp[num]
                else:
                    continue
    return dic
dic_construct(dic,'/Users/hhy/Desktop/1/512/lab.csv',1)
#dic_construct(dic,'/Users/hhy/Desktop/1/history.csv',1)
dic_construct(dic,'/Users/hhy/Desktop/1/512/herb.csv',0)
dic_construct(dic,'/Users/hhy/Desktop/1/512/pulse.csv',0)
dic_construct(dic,'/Users/hhy/Desktop/1/512/sym.csv',0)
dic_construct(fdic,'/Users/hhy/Desktop/1/512/flab.csv',1)
#dic_construct(fdic,'/Users/hhy/Desktop/1/fhistory.csv',1)
dic_construct(fdic,'/Users/hhy/Desktop/1/512/fherb.csv',0)
dic_construct(fdic,'/Users/hhy/Desktop/1/512/fpulse.csv',0)
dic_construct(fdic,'/Users/hhy/Desktop/1/512/fsym.csv',0)

#dic_construct(dic,'/Users/hhy/Desktop/1/admit.csv',0)
#dic_construct(fdic,'/Users/hhy/Desktop/1/fadmit.csv',0)

dicMerge={**dic,**fdic} #合并
res=[]  #所有特征集合
for num in dicMerge.values():
    for content in num.keys():
        if content not in res:
            res.append(content)
#print(res)
rownum,frownum,col=len(dic),len(fdic),len(res)
#tag
tag,ftag=np.ones(rownum),np.zeros(frownum)
#random_list=random.sample(range(0,frownum),rownum)
#print(len(random_list))
initial=np.zeros((rownum,col))
b=np.c_[initial,tag]
a=b.astype(float)
finitial=np.zeros((frownum,col))
d=np.c_[finitial,ftag]
c=d.astype(float)
#为保障字典的有序性，对keys进行排序，按照从低到高进行输出
dic_keys=sorted(dic.keys())
fdic_keys=sorted(fdic.keys())
for i in range(col):
    m=0
    for j in dic_keys:
        if res[i] in dic[str(j)].keys():
            a[m][i]=dic[str(j)][res[i]]
        m+=1
for i in range(col):
    m=0
    for j in fdic_keys:
        if res[i] in fdic[str(j)].keys():
            c[m][i]=fdic[str(j)][res[i]]
        m+=1
matrix=np.vstack((a,c))

#归一化
from sklearn import preprocessing
min_max_scaler = preprocessing.MinMaxScaler()
X_train_minmax = min_max_scaler.fit_transform(matrix)
data = pd.DataFrame(X_train_minmax)
#res.append('label')
data.to_csv('/Users/hhy/Desktop/1/512/test.csv',encoding='utf-8-sig',header=False,index=False)

feature=pd.DataFrame(res)
feature.to_csv('/Users/hhy/Desktop/1/512/feature.csv',encoding='utf-8-sig',header=False,index=False)
