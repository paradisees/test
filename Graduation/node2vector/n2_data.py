import csv
import pandas as pd
#处理成最后可输入矩阵
content={}
with open('/Users/hhy/Desktop/1/node/all/vector.csv','r',encoding='utf-8_sig') as f:
    csv_reader=csv.reader(f)
    for x in csv_reader:
        tmp=[]
        for i in range(1,len(x)):
            tmp.append(x[i])
        content[x[0]]=tmp
id={}
new=[]
with open('/Users/hhy/Desktop/1/node/all/label_new.csv','r',encoding='utf-8_sig') as f:
    csv_reader=csv.reader(f)
    for x in csv_reader:
        content[x[0]].append(x[1])
        new.append(content[x[0]])
data = pd.DataFrame(new)
data.to_csv('/Users/hhy/Desktop/1/node/all/matrix.csv',encoding='utf-8-sig',header=False,index=False)