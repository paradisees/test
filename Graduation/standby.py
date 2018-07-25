from sklearn.linear_model import (LinearRegression, Ridge,Lasso, RandomizedLasso)
from sklearn.feature_selection import RFE, f_classif
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import RidgeCV,LassoCV
import numpy as np
from minepy import MINE
import csv
import pandas as pd
data,mark,names=[],[],[]
init_fea={}
with open('/Users/hhy/Desktop/1/test.csv','r',encoding='utf-8_sig') as f:
    csv_reader=csv.reader(f)
    for x in csv_reader:
        data.append(list(map(float,x[0:-1])))
        mark.append(float(x[-1]))
with open('/Users/hhy/Desktop/1/feature.csv','r',encoding='utf-8_sig') as f:
    csv_reader=csv.reader(f)
    i=0
    for x in csv_reader:
        names.append(x[-1])
        init_fea[x[-1]]=i
        i+=1
#行为特征选择的算法，列为特征的名称
algorithm={}

def rank_to_dict(ranks, names, order=1):
    minmax = MinMaxScaler()
    ranks = minmax.fit_transform(order * np.array(ranks))
    ranks = map(lambda x: round(x, 5), ranks)
    return dict(zip(names, ranks))
#stability
rlasso = RandomizedLasso()
rlasso.fit(data, mark)
algorithm["stability"] = rank_to_dict(np.abs(rlasso.scores_), names)

#rf
rf = RandomForestClassifier()
rf.fit(data, mark)
algorithm["RF"] = rank_to_dict(rf.feature_importances_, names)

#GBDT
gbdt=GradientBoostingClassifier()
gbdt.fit(data, mark)
algorithm["GBDT"] = rank_to_dict(gbdt.feature_importances_, names)

#Extra
model = ExtraTreesClassifier()
model.fit(data, mark)
algorithm["Extra"] = rank_to_dict(model.feature_importances_, names)

#MIC
mine = MINE()
mic_scores = []
res=[]
for i in range(len(data[0])):
    for num in data:
        res.append(num[i])
    mine.compute_score(res, mark)
    m = mine.mic()
    mic_scores.append(m)
    res = []
algorithm["MIC"] = rank_to_dict(mic_scores, names)

#线性回归
lr = LinearRegression(normalize=True)
lr.fit(data, mark)
algorithm["Linear"] = rank_to_dict(np.abs(lr.coef_), names)

#ridge
ridgecv = RidgeCV()
ridgecv.fit(data, mark)
#print(ridgecv.alpha_)
ridge = Ridge(alpha=ridgecv.alpha_)
ridge.fit(data, mark)
algorithm["Ridge"] = rank_to_dict(np.abs(ridge.coef_), names)

#lasso
lassocv = LassoCV()
lassocv.fit(data, mark)
#print(lassocv.alpha_)
lasso = Lasso(alpha=lassocv.alpha_)
lasso.fit(data, mark)
algorithm["Lasso"] = rank_to_dict(np.abs(lasso.coef_), names)

#rfe
log=LogisticRegression()
rfe = RFE(log, n_features_to_select=10)
rfe.fit(data, mark)
algorithm["RFE"] = rank_to_dict(list(map(float, rfe.ranking_)), names, order=-1)
'''
#f值检验
f, pval = f_classif(data, mark)
algorithm["Corr"] = rank_to_dict(f, names)
'''
r = {}
for name in names:
    r[name] = round(np.mean([algorithm[method][name] for method in algorithm.keys()]), 4)
methods = sorted(algorithm.keys())
algorithm["Mean"] = r
methods.append("Mean")

content=[]
for name in names:
    content.append([algorithm[method][name] for method in methods])
fea_matrix = pd.DataFrame(content,index=names)
#fea_matrix.to_csv('/Users/hhy/Desktop/test/fea_importance.csv',encoding='utf-8-sig',header=methods)


label=[]
for num in mark:
    label.append([num])
'''
def score(data,flag,mark=label):
    from sklearn import cross_validation
    from sklearn.ensemble import RandomForestClassifier
    from sklearn import svm
    res=[]
    for i in range(10):
        X_train, X_test, y_train, y_test = cross_validation.train_test_split(
        data, mark, test_size=0.1, random_state=i)
        if flag=='svm':
            clf = svm.SVC(kernel='linear', C=1)
        elif flag=='rf':
            clf = RandomForestClassifier(n_estimators= 600)
        elif flag=='lr':
            clf = LogisticRegression()
        clf.fit(X_train, y_train)
        #print('准确率:',clf.score(X_test, y_test))
        res.append(clf.score(X_test, y_test))
    return (sum(res)/len(res))
'''
'''倒序选择重要的特征名，feature_count表示需要的特征个数'''

fea={}
for name in names:
    fea[name]=algorithm['Mean'][name]
new=sorted(fea.items(),key=lambda x:x[1],reverse=True)

def n_feature(feature_count):
    feature = []  # 特征名集合
    i=0
    for num in new:
        if i<feature_count:
            feature.append(num[0])
            i+=1
        else:
            break
    feature.append('label')
    return feature
def modify_data(feature):
    new_data = []  # 修改原数据为所需特征对应的数据
    for num in data:
        tmp=[]
        for content in feature:
            if content in init_fea.keys():
                tmp.append(num[init_fea[content]])
        new_data.append(tmp)
    return new_data
'''
#算出选取多少个特征合适
feature_count=[i for i in range(5,251,5)]
id,tmp=0,0
for content in feature_count:
    feature=n_feature(content)
    new_data=modify_data(feature)
    Score=(score(new_data,'lr')+score(new_data,'svm')+score(new_data,'rf'))/3
    if Score>id:
        id=Score
        tmp=content
print('得分较高：',[tmp,id])
'''
from Graduation.service import forward
content=[]
t1=0
t2=[]
for tmp in range(50,201,2):
    feature = n_feature(tmp)
    new_data = modify_data(feature)
    auc=forward(new_data,label)
    auc=round(auc,5)
    content.append((auc,tmp))
print(sorted(content,reverse=True))
'''
    if auc>=t1:
        t1=auc
        t2=new_data
        number=tmp
matrix=np.hstack((t2,label))
Data = pd.DataFrame(matrix)
Data.to_csv('/Users/hhy/Desktop/test/Data'+str(number)+'.csv',encoding='utf-8-sig',header=False,index=False)

'''

'''
fea_select=pd.DataFrame(feature)
fea_select.to_csv('/Users/hhy/Desktop/test/fea_select.csv',encoding='utf-8-sig',header=False,index=False)
'''