import csv
import numpy as np
import pandas as pd
def fea(text):
    import csv
    with open(text) as f:
        rows = csv.reader(f)
        res=[]
        for row in rows:
            for i in range(len(row)):
                if row[i]!='':
                    res.append(row[i])
        fea = list(set(res))
        return fea
res=fea('/Users/hhy/Desktop/1/538.csv')
#print(res,len(res))

items=[]
with open('/Users/hhy/Desktop/1/test.csv') as f:
    rows = csv.reader(f)
    for row in rows:
        items.append(row)
str=[]
for i in range(len(items[0])):
    if items[0][i] not in res:
        str.append(i)
#print(str,len(str))

for item in items:
    j=0
    for i in str:
        item.pop(i-j)
        j+=1
print(len(items[0]))

data1 = pd.DataFrame(items)
print(len(data1))
data1.to_csv('/Users/hhy/Desktop/1/222.csv')