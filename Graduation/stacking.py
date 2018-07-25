from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import StratifiedKFold
import numpy as np
from sklearn.metrics import roc_auc_score
import xgboost as xgb
from sklearn import cross_validation,metrics
import csv
from sklearn import svm
from sklearn import cross_validation
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model.logistic import LogisticRegression
liquence=[0,3,7,8]
for i in liquence:
    data=[]
    mark=[]
    with open('/Users/hhy/Desktop/data.csv','r',encoding='utf-8_sig') as f1:
        csv_reader=csv.reader(f1)
        for x in csv_reader:
            data.append(list(map(float,x[0:-1])))
            mark.append(list(map(float,x[-1])))
    X=np.array(data)
    #y=mark
    target=[]
    [target.extend(i) for i in mark]
    target=np.array(target)
    '''模型融合中使用到的各个单模型'''
    clfs = [LogisticRegression(C=4.8,random_state=1113),
            svm.SVC(kernel='linear', C=3,probability=True,gamma=0.001,random_state=1113),
            GradientBoostingClassifier(learning_rate=0.03, random_state=1113, n_estimators=1100, min_samples_split=28,min_samples_leaf=4, max_depth=5, max_features=9, subsample=0.8),
            GradientBoostingClassifier(loss='exponential', learning_rate=0.04, n_estimators=1000, max_depth=7,min_samples_split=48, min_samples_leaf=6, max_features=9, subsample=0.7,random_state=1113),
            xgb.XGBClassifier(n_estimators=900, learning_rate=0.1, gamma=0.2, subsample=0.8, max_depth=10,colsample_bytree=0.8, objective='binary:logistic', seed=1113),
            xgb.XGBClassifier(n_estimators=900, learning_rate=0.1, gamma=0.2, subsample=0.8, max_depth=10,colsample_bytree=0.8, objective='binary:logistic', seed=1113),
            RandomForestClassifier(random_state=1113, n_estimators=900, max_depth=24, max_features=8, min_samples_split=2),
            ExtraTreesClassifier(random_state=1113, n_estimators=1800, max_depth=28, max_features=15),
            #MLPClassifier(random_state=1113)
            ]
    '''切分一部分数据作为测试集'''
    #X-y（训练集）   X_predict-y_predict（测试集）
    X, X_predict, y, y_predict = train_test_split(X, target, test_size=0.05, random_state=i)
    #第二层模型的训练和测试集
    dataset_blend_train = np.zeros((X.shape[0], len(clfs)))  #1265*2
    dataset_blend_test = np.zeros((X_predict.shape[0], len(clfs))) #141*2

    '''10折stacking'''
    n_folds = 5
    #切分训练集标签长度为十份，做交叉验证
    skf = list(StratifiedKFold(y, n_folds))  #总长度为10，每部分分为训练和测试的下标号
    for j, clf in enumerate(clfs):
        '''依次训练各个单模型'''
        dataset_blend_test_j = np.zeros((X_predict.shape[0], len(skf))) #141*10
        for i, (train, test) in enumerate(skf):
            '''使用第i个部分作为预测，剩余的部分来训练模型，获得其预测的输出作为第i部分的新特征。'''
            X_train, y_train, X_test, y_test = X[train], y[train], X[test], y[test]
            clf.fit(X_train, y_train)
            y_submission = clf.predict_proba(X_test)[:, 1]
            dataset_blend_train[test, j] = y_submission
            #print(clf.predict_proba(X_predict)[:, 1])
            dataset_blend_test_j[:, i] = clf.predict_proba(X_predict)[:, 1]
        '''对于测试集，直接用这k个模型的预测值均值作为新的特征。'''
        dataset_blend_test[:, j] = dataset_blend_test_j.mean(1)
        #print("val auc Score: %f" % roc_auc_score(y_predict, dataset_blend_test[:, j]))
    #for clf in clfs:
    clf = LogisticRegression(C=4.8,random_state=1113)
    #clf = svm.SVC(kernel='linear', C=2)
    clf.fit(dataset_blend_train, y)
    y_submission = clf.predict_proba(dataset_blend_test)[:, 1]
    #y_submission = (y_submission - y_submission.min()) / (y_submission.max() - y_submission.min())
    print("blend result val auc Score: %f" % (roc_auc_score(y_predict, y_submission)))
    y_pred = clf.predict(dataset_blend_test)
    print("blend result val acc Score: %f" % (metrics.accuracy_score(y_predict, y_pred)))
'''0.891534
0.886243
0.887125
0.888007
'''