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
data=[]
mark=[]
with open('/Users/hhy/Desktop/1/test.csv','r',encoding='utf-8_sig') as f:
    csv_reader=csv.reader(f)
    for x in csv_reader:
        data.append(list(map(float,x[0:-1])))
        mark.append(float(x[-1]))
auc=[]
acc=[]
f1=[]
for i in range(10):
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(
    data, mark, test_size=0.05, random_state=i)
    clf = LogisticRegression(C=4.8,random_state=1113)
    clf.fit(X_train, y_train)
    y_predict = clf.predict_proba(X_test)[:, 1]
    test_auc = metrics.roc_auc_score(y_test, y_predict)  # 验证集上的auc值
    auc.append(test_auc)
    y_pred = clf.predict(X_test)
    acc.append(metrics.accuracy_score(y_test, y_pred))
    f1.append(metrics.f1_score(y_test, y_pred))
print("acc==", sum(acc) / len(acc))
print("auc==", sum(auc) / len(auc))
print("f1==", sum(f1) / len(f1))
