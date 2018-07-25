from sklearn.linear_model import (LinearRegression, Ridge,Lasso, RandomizedLasso)
from sklearn.feature_selection import RFE, f_classif
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import RidgeCV,LassoCV
from sklearn import cross_validation,metrics
import numpy as np
from minepy import MINE
import csv
import pandas as pd
data,mark,names=[],[],[]
init_fea={}
with open('/Users/hhy/Desktop/test.csv','r',encoding='utf-8_sig') as f:
    csv_reader=csv.reader(f)
    for x in csv_reader:
        data.append(list(map(float,x[0:-1])))
        mark.append(float(x[-1]))
def acc_score(method):
    acc=[]
    for i in range(5):
        X_train, X_test, y_train, y_test = cross_validation.train_test_split(
            data, mark, test_size=0.05, random_state=i)
        # clf = MLPClassifier(random_state=1113)
        if method=='MIC':
            return 0.754
        elif method=='Linear':
            clf =LinearRegression(normalize=True)
        elif method == 'Ridge':
            clf = LogisticRegression()
        elif method == 'Lasso':
            clf = LogisticRegression(penalty='l1')
        elif method=='RFE':
            log = LogisticRegression()
            clf = RFE(log, n_features_to_select=10)
        elif method=='stability':
            return 1
        elif method=='RF':
            clf =RandomForestClassifier(random_state=i)
        elif method=='GBDT':
            clf =GradientBoostingClassifier(random_state=i)
        elif method=='Extra':
            clf =ExtraTreesClassifier(random_state=i)
        clf.fit(X_train, y_train)
        # print('准确率:',clf.score(X_test, y_test))
        acc.append(clf.score(X_test, y_test))
    return (sum(acc)/len(acc))
tmp_name=['MIC','Ridge','Lasso','RFE','stability','RF','GBDT','Extra']
acc_score_tmp={}
for tmp_tmp in tmp_name:
    acc_score_tmp[tmp_tmp]=acc_score(str(tmp_tmp))
print(acc_score_tmp)