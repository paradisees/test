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
with open('/Users/hhy/Desktop/test.csv','r',encoding='utf-8_sig') as f:
    csv_reader=csv.reader(f)
    for x in csv_reader:
        data.append(list(map(float,x[0:-1])))
        mark.append(float(x[-1]))
with open('/Users/hhy/Desktop/feature.csv','r',encoding='utf-8_sig') as f:
    csv_reader=csv.reader(f)
    i=0
    for x in csv_reader:
        names.append(x[-1])
        init_fea[x[-1]]=i
        i+=1
def rank_to_dict(ranks, names, order=1, cv=False):
    minmax = MinMaxScaler()
    ranks = minmax.fit_transform(order * np.array(ranks))
    if cv==True:
        ranks/=10
    ranks = map(lambda x: round(x, 5), ranks)
    return dict(zip(names, ranks))
#合并字典
def add(x,y):
    for k, v in y.items():
        if k in x.keys():
            x[k] += v
        else:
            x[k] = v
    return x
def score_calculate(flag):
    # 行为特征选择的算法，列为特征的名称
    algorithm = {}
    if flag=='whole':
        tmp_sta,tmp_rf,tmp_gbdt,tmp_extra={},{},{},{}
        for n in range(10):
            #stability
            rlasso = RandomizedLasso(random_state=n)
            rlasso.fit(data, mark)
            tmp_sta = add(tmp_sta,rank_to_dict(np.abs(rlasso.scores_), names,cv=True))

            #rf
            rf = RandomForestClassifier(random_state=n)
            rf.fit(data, mark)
            tmp_rf = add(tmp_rf,rank_to_dict(rf.feature_importances_, names,cv=True))

            #GBDT
            gbdt=GradientBoostingClassifier(random_state=n)
            gbdt.fit(data, mark)
            tmp_gbdt = add(tmp_gbdt, rank_to_dict(gbdt.feature_importances_, names, cv=True))

            #Extra
            model = ExtraTreesClassifier(random_state=n)
            model.fit(data, mark)
            tmp_extra = add(tmp_extra, rank_to_dict(model.feature_importances_, names, cv=True))

        algorithm["stability"],algorithm["RF"],algorithm["GBDT"],algorithm["Extra"] \
            = tmp_sta,tmp_rf,tmp_gbdt,tmp_extra
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
    elif flag=='extra':
        model = ExtraTreesClassifier()
        model.fit(data, mark)
        algorithm["Extra"] = rank_to_dict(model.feature_importances_, names)
    elif flag=='gbdt':
        gbdt = GradientBoostingClassifier()
        gbdt.fit(data, mark)
        algorithm["GBDT"] = rank_to_dict(gbdt.feature_importances_, names)
    elif flag=='rf':
        rf = RandomForestClassifier()
        rf.fit(data, mark)
        algorithm["RF"] = rank_to_dict(rf.feature_importances_, names)
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
    fea_matrix.to_csv('/Users/hhy/Desktop/fea_importance_'+flag+'.csv',encoding='utf-8-sig',header=methods)
    return algorithm
algorithm=score_calculate('whole')
label=[]
for num in mark:
    label.append([num])

def score(data,flag,mark=label):
    from sklearn import cross_validation
    from sklearn.ensemble import RandomForestClassifier
    from sklearn import svm
    from sklearn.ensemble import ExtraTreesClassifier
    from sklearn.ensemble import GradientBoostingClassifier

    res=[]
    for i in range(10):
        X_train, X_test, y_train, y_test = cross_validation.train_test_split(
        data, mark, test_size=0.05, random_state=i)
        if flag=='svm':
            clf = svm.SVC(kernel='linear', C=2,random_state=1113)
        elif flag=='rf':
            clf = RandomForestClassifier(random_state=1113)
        elif flag=='lr':
            clf = LogisticRegression(C=4.8,random_state=1113)
        elif flag=='extra':
            clf = ExtraTreesClassifier(random_state=1113)
        elif flag=='gbdt':
            clf = GradientBoostingClassifier(random_state=1113)
        elif flag=='adaboost':
            clf = GradientBoostingClassifier(loss='exponential',random_state=1113)
        clf.fit(X_train, y_train)
        #print('准确率:',clf.score(X_test, y_test))
        res.append(clf.score(X_test, y_test))
    return (sum(res)/len(res))


'''倒序选择重要的特征名，feature_count表示需要的特征个数'''
fea={}
for name in names:
    fea[name]=algorithm['Mean'][name]
new=sorted(fea.items(),key=lambda x:(x[1],x[0]),reverse=True)

'''
#调用test检测forward多次是否为同一结果
from Graduation.service import test_forward
new=test_forward()
'''
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

#算出选取多少个特征合适
feature_count=[i for i in range(50,201,2)]
id,tmp=0,0    #tmp为特征个数
for content in feature_count:
    feature=n_feature(content)
    new_data=modify_data(feature)
    Score=score(new_data,'lr')
    if Score>id:
        id=Score
        tmp=content
print('得分较高：',[tmp,id])
feature = n_feature(tmp)
new_data = modify_data(feature)

'''
stacking=['lr','svm','rf','extra','gbdt','adaboost']
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
#0.7704
#输出最好特征个数的矩阵

matrix=np.hstack((new_data,label))
Data = pd.DataFrame(matrix)
Data.to_csv('/Users/hhy/Desktop/Data'+str(tmp)+'.csv',encoding='utf-8-sig',header=False,index=False)

'''
#输出所选特征的名称
fea_select=pd.DataFrame(feature)
fea_select.to_csv('/Users/hhy/Desktop/1/node/fea_select.csv',encoding='utf-8-sig',header=False,index=False)
'''