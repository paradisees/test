import pandas as pd
import csv
from sklearn.ensemble import AdaBoostClassifier,GradientBoostingClassifier
import xgboost as xgb
data=[]
with open('/Users/hhy/Desktop/tmp.csv','r',encoding='utf-8_sig') as f:
    csv_reader=csv.reader(f)
    for x in csv_reader:
        data.append(list(map(float,x[:])))

for i in range(len(data)):
    for j in range(len(data[i])):
        data[i][j]=round(data[i][j],2)
print(data)

Data = pd.DataFrame(data)
Data.to_csv('/Users/hhy/Desktop/tmp1.csv',encoding='utf-8-sig',header=False,index=False)

