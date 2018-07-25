#MIC互信息
from minepy import MINE
import numpy as np
from sklearn import cross_validation
m = MINE()
import csv
data=[]
mark=[]
with open('/Users/hhy/Desktop/test.csv','r',encoding='utf-8_sig') as f:
    csv_reader=csv.reader(f)
    for x in csv_reader:
        data.append(list(map(int,x[0:-1])))
        mark.append(int(x[-1]))
res=[]
dic={}
for i in range(len(data[0])):
    for num in data:
        res.append(num[i])
    m.compute_score(res, mark)
    dic[i]=m.mic()
    m=MINE()
    res=[]
new=sorted(dic.items(),key=lambda x:x[1])
print(new)



#稳定性选择
from sklearn.linear_model import RandomizedLasso
import csv
data=[]
mark=[]
name=[]
with open('/Users/hhy/Desktop/test.csv','r',encoding='utf-8_sig') as f:
    csv_reader=csv.reader(f)
    for x in csv_reader:
        data.append(list(map(int,x[0:-1])))
        mark.append(int(x[-1]))
with open('/Users/hhy/Desktop/feature.csv','r',encoding='utf-8_sig') as f:
    csv_reader=csv.reader(f)
    for x in csv_reader:
        name.append(x[-1])
rlasso = RandomizedLasso()
rlasso.fit(data, mark)
print("Features sorted by their score:")
print(sorted(zip(map(lambda x: round(x, 4), rlasso.scores_),name),
                  reverse=True))




#rfe
from sklearn import cross_validation
from sklearn.linear_model import LinearRegression
from sklearn.linear_model.logistic import LogisticRegression
import csv
data=[]
mark=[]
name=[]
with open('/Users/hhy/Desktop/test.csv','r',encoding='utf-8_sig') as f:
    csv_reader=csv.reader(f)
    for x in csv_reader:
        data.append(list(map(int,x[0:-1])))
        mark.append(int(x[-1]))
with open('/Users/hhy/Desktop/feature.csv','r',encoding='utf-8_sig') as f:
    csv_reader=csv.reader(f)
    for x in csv_reader:
        name.append(x[-1])
from sklearn.svm import SVC
from sklearn.feature_selection import RFE
import matplotlib.pyplot as plt
X_train, X_test, y_train, y_test = cross_validation.train_test_split(
    data, mark, test_size=0.1)
'''svc = SVC(kernel="linear", C=1)
rfe = RFE(estimator=svc, n_features_to_select=1, step=1)'''
lr = LogisticRegression()
rfe = RFE(lr, n_features_to_select=10)
rfe.fit(X_train, y_train)
#print(len(rfe.ranking_),'--',rfe.support_,'--',rfe.n_features_)
print ("Features sorted by their rank:")
print (sorted(zip(map(lambda x: round(x, 4), rfe.ranking_), name)))
