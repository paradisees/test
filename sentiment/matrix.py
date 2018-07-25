#构建矩阵
import csv
import numpy as np
import pandas as pd
def fea(text):
    import csv
    with open(text) as f:
        rows = csv.reader(f)
        res=[]
        num=0
        for row in rows:
            num+=1
            for i in range(len(row)):
                if row[i]!='':
                    res.append(row[i])
        fea = list(set(res))
        return fea,num
res,rownum=fea('/Users/hhy/Desktop/2/test.csv')
col=len(res)
with open('/Users/hhy/Desktop/2/fea.csv', 'w', encoding='gb18030', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(res)

dic={}

#构建空字典
for i in range(rownum):
    dic[str(i)]=[]
with open('/Users/hhy/Desktop/2/test.csv') as csvfile1:
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
data1.to_csv('/Users/hhy/Desktop/2/newtest.csv')
