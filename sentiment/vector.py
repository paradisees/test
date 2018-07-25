#词向量生成特征矩阵
import csv
import numpy as np
import pandas as pd
with open('/Users/hhy/Desktop/3/cat500.csv','r',encoding='gbk') as f1:
    rows=csv.reader(f1)
    dic={}
    num=500
    for i in range(num):
        dic[str(i)] = []
    for row in rows:
        dic[str(row[1])].append(str(row[0]))
#print(dic)
with open('/Users/hhy/Desktop/3/participle.csv','r',encoding='gbk') as f1:
    rows=csv.reader(f1)
    rownum=0
    for row in rows:
        rownum+=1
    b=np.zeros((rownum,num))
    a=b.astype(int)
with open('/Users/hhy/Desktop/3/participle.csv', 'r', encoding='gbk') as f1:
    rows = csv.reader(f1)
    n = 0
    for row in rows:
        for i in range(len(row)):
            for j in range(num):
                if row[i]!='' and row[i] in dic[str(j)]:
                    a[n][j]+=1
        n+=1
data1 = pd.DataFrame(a)
data1.to_csv('/Users/hhy/Desktop/3/vectortrain500.csv')

with open('/Users/hhy/Desktop/3/test.csv','r',encoding='gbk') as f:
    rows1=csv.reader(f)
    rownum1=0
    for row in rows1:
        rownum1+=1
    d=np.zeros((rownum1,num))
    c=d.astype(int)
with open('/Users/hhy/Desktop/3/test.csv', 'r', encoding='gbk') as f:
    rows1 = csv.reader(f)
    n = 0
    for row in rows1:
        for i in range(len(row)):
            for j in range(num):
                if row[i]!='' and row[i] in dic[str(j)]:
                    c[n][j]+=1
        n+=1
data2 = pd.DataFrame(c)
data2.to_csv('/Users/hhy/Desktop/3/vectortest500.csv')