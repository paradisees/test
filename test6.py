import numpy as np
from sklearn.lda import LDA
from sklearn import cross_validation,metrics
from sklearn.linear_model import (LinearRegression, Ridge,Lasso, RandomizedLasso)
from sklearn import svm
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.linear_model import RidgeCV,LassoCV
from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier
from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.cross_validation import StratifiedKFold
import csv
from sklearn.ensemble import AdaBoostClassifier,GradientBoostingClassifier
import xgboost as xgb
data=[]
mark=[]
with open('/Users/hhy/Desktop/test/data.csv','r',encoding='utf-8_sig') as f:
    csv_reader=csv.reader(f)
    for x in csv_reader:
        data.append(list(map(float,x[0:-1])))
        mark.append(float(x[-1]))
#number=[0,1,2,3,11,12,14,16,17,18]
acc=[]
auc=[]
f1=[]
for i in range(10):
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(
    data, mark, test_size=0.05, random_state=i)
    clf1 = LogisticRegression(C=4.8,random_state=1113)  #0.8812
    clf1.fit(X_train, y_train)

    #print('准确率:',clf.score(X_test, y_test))
    acc.append(clf1.score(X_test, y_test))

    y_predict = clf1.predict_proba(X_test)[:,1]
    test_auc = metrics.roc_auc_score(y_test, y_predict)  # 验证集上的auc值
    #print('AUC:', test_auc)
    auc.append(test_auc)
    #print('F1值:',metrics.f1_score(y_test, y_predict))
    y_pred = clf1.predict(X_test)
    f1.append(metrics.f1_score(y_test, y_pred))
print("==",sum(acc)/len(acc))
print("==",sum(auc)/len(auc))
print("==",sum(f1)/len(f1))
#print("==",sum(train)/len(train))
