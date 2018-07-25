import numpy as np
from sklearn.lda import LDA
from sklearn import cross_validation,metrics
from sklearn.linear_model import (LinearRegression, Ridge,Lasso, RandomizedLasso)
from sklearn import svm
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.linear_model import RidgeCV,LassoCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
import csv
'''
#输入原始数据集上的准确率和auc作为比对
data = []
mark = []
with open('/Users/hhy/Desktop/1/herb_only/test.csv','r',encoding='utf-8_sig') as f:
    csv_reader=csv.reader(f)
    for x in csv_reader:
        data.append(list(map(float,x[0:-1])))
        mark.append(float(x[-1]))

acc=[]
auc=[]
tmp_acc = []
tmp_auc = []
for i in range(10):
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(
    data, mark, test_size=0.05, random_state=i)
    clf = LogisticRegression(C=4.8,random_state=1113)
    clf.fit(X_train, y_train)
    tmp_acc.append(clf.score(X_test, y_test))
    y_predict = clf.predict_proba(X_test)[:, 1]
    tmp_auc.append(metrics.roc_auc_score(y_test, y_predict))  # 验证集上的auc值
acc.append(sum(tmp_acc)/len(tmp_acc))
auc.append(sum(tmp_auc)/len(tmp_auc))
print('original acc:',acc)
print('original auc:',auc)
'''
#set=[0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9]
#set=[0.26,0.27,0.28,0.29,0.3,0.31,0.32,0.33,0.34]
set=[0.25,0.3,0.35,0.4,0.38,0.42,0.45,0.48,0.5,0.55]
number=[8, 12, 16, 20, 24, 28,32,64,96,128,160,192,256,288,352,416,480,512]

final_acc={}
final_auc={}
for threshold in set:
    for num in number:
        data = []
        mark = []
        #with open('/Users/hhy/Desktop/1/node/final/cos/herb_only_meansum_'+str(threshold)+'.csv','r',encoding='utf-8_sig') as f:
        with open('/Users/hhy/Desktop/1/node/final/cmp/'+str(num)+'emb'+str(threshold)+'.csv', 'r',encoding='utf-8_sig') as f:
            csv_reader=csv.reader(f)
            for x in csv_reader:
                data.append(list(map(float,x[0:-1])))
                mark.append(float(x[-1]))
        tmp_acc = []
        tmp_auc = []
        name=str(num)+'emb'+str(threshold)
        for i in range(10):
            X_train, X_test, y_train, y_test = cross_validation.train_test_split(
            data, mark, test_size=0.05, random_state=i)
            clf = LogisticRegression(C=4.8,random_state=1113)
            clf.fit(X_train, y_train)
            tmp_acc.append(clf.score(X_test, y_test))
            y_predict = clf.predict_proba(X_test)[:, 1]
            tmp_auc.append(metrics.roc_auc_score(y_test, y_predict))  # 验证集上的auc值
        final_acc[name]=(sum(tmp_acc)/len(tmp_acc))
        final_auc[name]=(sum(tmp_auc)/len(tmp_auc))
final_acc=sorted(final_acc.items(),key=lambda x:x[1],reverse=True)
final_auc=sorted(final_auc.items(),key=lambda x:x[1],reverse=True)
print('final acc:',final_acc)
print('final auc:',final_auc)

