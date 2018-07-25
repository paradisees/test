import pandas as pd
import numpy as np
from sklearn.externals import joblib
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import confusion_matrix
from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier
from sklearn.grid_search import GridSearchCV
import csv
data=[]
mark=[]
with open('/Users/hhy/Desktop/test.csv','r',encoding='utf-8_sig') as f:
    csv_reader=csv.reader(f)
    for x in csv_reader:
        data.append(list(map(int,x[0:-1])))
        mark.append(int(x[-1]))
#param_test1 = {'n_estimators':[i for i in range(900,3000,100)]}
#param_test2 = {'max_depth':[i for i in range(3,14,1)]}
#param_test3 = {'min_samples_split':[i for i in range(5,30,5)], 'min_samples_leaf':[j for j in range(2,11,1)]}
#param_test4 = {'max_features':[i for i in range(10,313,20)]}
#param_test5 = {'subsample':[0.6,0.7,0.75,0.8,0.85,0.9]}
gbdt=GradientBoostingClassifier(
    loss='deviance',
    #loss='exponential'
    learning_rate=0.05,
    n_estimators=1900,
    subsample=0.8,
    max_features=70,
    max_depth=7,
    #verbose=1
)

'''
gbdt = GridSearchCV(estimator = GradientBoostingClassifier(learning_rate=0.05,max_depth=7,max_features=70,subsample=0.8,random_state=10,n_estimators=1900),
param_grid = param_test5, scoring='roc_auc',n_jobs=4,iid=False, cv=5)
'''
res=[] #准确率
fea1=[] #多组特征构成的集合，特征筛选 >0.005
fea2=[] #>0.001
for i in range(5):
    tmp1,tmp2=[],[]
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(
    data, mark, test_size=0.1)
    clf=gbdt.fit(X_train,y_train)
#print(clf.grid_scores_,clf.best_params_,clf.best_score_)
    print(clf.score(X_test, y_test)) #预测准确率
#print(gbdt.feature_importances_)
    res.append(clf.score(X_test, y_test))
#joblib.dump(clf,'/Users/hhy/Desktop/3/judge.m')
#print(sum(res)/10)
    for j in range(len(gbdt.feature_importances_)):
        if gbdt.feature_importances_[j]>0.005:
            tmp1.append(j)
        if gbdt.feature_importances_[j]>0.001:
            tmp2.append(j)
    fea1.append(tmp1)
    fea2.append(tmp2)
print(fea1)
'''处理交集特征'''
from functools import reduce
feature1=list(reduce(lambda x,y : set(x) & set(y), fea1))#序号
feature2=list(reduce(lambda x,y : set(x) & set(y), fea2))
print(feature1,len(feature1))
#print(feature2,len(feature2))


'''查看具体筛选特征'''
feature=[]
with open('/Users/hhy/Desktop/feature.csv','r',encoding='utf-8_sig') as f1:
    csv_reader=csv.reader(f1)
    i=0
    for x in csv_reader:
        if i in feature1: #选择看哪类特征
        #if i in feature2:  # 选择看哪类特征
            feature.append(x[0])
        i+=1
feature.append('label') #增加最后一列类别的标签
print(feature)

'''删除非筛选特征列'''
for num in data:
    j = 0
    for i in range(len(num)):
        if i not in feature1:
        #if i not in feature2:
            del(num[i - j])
            j+=1
label=[]
for num in mark:
    label.append([num])
matrix=np.hstack((data,label))
#print(matrix)
datawithgbdt = pd.DataFrame(matrix)
datawithgbdt.to_csv('/Users/hhy/Desktop/datawithgbdt1.csv',encoding='utf-8-sig',header=feature,index=False)