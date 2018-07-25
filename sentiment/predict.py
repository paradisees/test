#构建好的特征来预测需要的数据
import csv
import numpy as np
import pandas as pd
def fea(text):
    import csv
    with open(text,encoding='utf-8') as f:
        rows = csv.reader(f)
        res=[]
        for row in rows:
            for i in range(len(row)):
                if row[i]!='':
                    res.append(row[i])
        fea = list(set(res))
        return fea
#输入所需特征的那个csv
res=fea('/Users/hhy/Desktop/3/participle.csv')
col=len(res)
rownum=759
dic={}

#构建空字典
for i in range(rownum):
    dic[str(i)]=[]
#需要预测的数据
with open('/Users/hhy/Desktop/3/test.csv',encoding='utf-8') as csvfile1:
    rows = csv.reader(csvfile1)
    i=0
    #按序号在字典中添加词
    for row in rows:
        if i <= rownum:
            for j in range(len(row)):
                if row[j]!='':
                    dic[str(i)].append(str(row[j]))
            i+=1
#print(dic)

#构建矩阵
b=np.zeros((rownum,col))
a=b.astype(int)
for i in range(len(res)):
    for j in range(rownum):
        if res[i] in dic[str(j)]:
            a[j][i]=1
print(len(a))
data1 = pd.DataFrame(a)
print(len(data1))
data1.to_csv('/Users/hhy/Desktop/3/newtest.csv')
