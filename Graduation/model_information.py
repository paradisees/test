import numpy as np
from sklearn import cross_validation,metrics
from sklearn.ensemble import AdaBoostClassifier,GradientBoostingClassifier
import xgboost as xgb
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier
from sklearn import svm
import time
from sklearn.metrics import roc_curve, auc
import csv
import pandas as pd
data=[]
mark=[]
with open('/Users/hhy/Desktop/data.csv','r',encoding='utf-8_sig') as f:
    csv_reader=csv.reader(f)
    for x in csv_reader:
        data.append(list(map(float,x[0:-1])))
        mark.append(float(x[-1]))
data=np.array(data)
mark=np.array(mark)
clf1 = LogisticRegression(C=4.8,random_state=1113)  #0.8812
clf2 = svm.SVC(kernel='linear', C=3,probability=True,gamma=0.001,random_state=1113) #0.8824
clf3 = GradientBoostingClassifier(learning_rate=0.03,random_state=1113,n_estimators=1100,min_samples_split=28,min_samples_leaf=4,max_depth=5,max_features=9,subsample=0.8)  #0.8673
clf4 = GradientBoostingClassifier(loss='exponential', learning_rate=0.04, n_estimators=1000, max_depth=7,min_samples_split=48, min_samples_leaf=6, max_features=9, subsample=0.7,random_state=1113)   #adaboost  0.8612
clf5 = xgb.XGBClassifier(n_estimators=900,learning_rate =0.1,gamma=0.2, subsample=0.8,max_depth=10,colsample_bytree=0.8,objective= 'binary:logistic', seed=1113)
clf6 = RandomForestClassifier(random_state=1113,n_estimators=900,max_depth=24,max_features=8,min_samples_split=2)  #0.8443
clf7 = ExtraTreesClassifier(random_state=1113, n_estimators=1800, max_depth=28, max_features=15) #0.8538
header=['logistic','svm','gbdt','adaboost','xgboost','rf','extra']
clfs=[clf1,clf2,clf3,clf4,clf5,clf6,clf7]
auc,acc,f1,used_time=[],[],[],[]
for clf in clfs:
    tmp_auc=[]
    tmp_acc=[]
    tmp_f1=[]
    initial_time=time.time()
    for i in range(5):
        X_train, X_test, y_train, y_test = cross_validation.train_test_split(
        data, mark, test_size=0.05, random_state=i)
        clf.fit(X_train, y_train)
        y_predict = clf.predict_proba(X_test)[:,1]
        test_auc = metrics.roc_auc_score(y_test, y_predict)  # 验证集上的auc值
        #print('AUC:', test_auc)
        tmp_auc.append(test_auc)
        y_pred = clf.predict(X_test)
        #print ('ACC: %.4f' % metrics.accuracy_score(y_test,y_pred))
        tmp_acc.append(metrics.accuracy_score(y_test,y_pred))
        #print ('F1-score: %.4f' %metrics.f1_score(y_test,y_predict))
        tmp_f1.append(metrics.f1_score(y_test, y_pred))
    over_time=time.time()
    auc.append(round(sum(tmp_auc)/len(tmp_auc),3))
    acc.append(round(sum(tmp_acc)/len(tmp_acc),3))
    f1.append(round(sum(tmp_f1)/len(tmp_f1),3))
    used_time.append(round(over_time-initial_time,3))
index=['AUC','ACC','F1','time']
out=[]
out.append(auc)
out.append(acc)
out.append(f1)
out.append(used_time)
data = pd.DataFrame(out,index=index)
data.to_csv('/Users/hhy/Desktop/node_model_information.csv',encoding='utf-8-sig',header=header)
