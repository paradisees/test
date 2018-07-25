#同时计算多个test的准确率并写入csv
from sklearn import cross_validation
from sklearn.linear_model.logistic import LogisticRegression
import csv
import pandas as pd
from sklearn import cross_validation,metrics

tmp=[]
def score(id):
    data = []
    mark = []
    with open(id, 'r', encoding='utf-8_sig') as f:
        csv_reader = csv.reader(f)
        for x in csv_reader:
            data.append(list(map(float, x[0:-1])))
            mark.append(float(x[-1]))
    acc = []
    auc = []
    f1 = []
    for i in range(10):
        X_train, X_test, y_train, y_test = cross_validation.train_test_split(
            data, mark, test_size=0.05, random_state=i)
        clf = LogisticRegression(C=4.8, random_state=1113)
        clf.fit(X_train, y_train)
        # print('准确率:',clf.score(X_test, y_test))
        acc.append(round(clf.score(X_test, y_test), 3))
        y_pred = clf.predict(X_test)
        # print ('ACC: %.4f' % metrics.accuracy_score(y_test,y_pred))
        auc.append(round(metrics.roc_auc_score(y_test, y_pred),3))
        # print ('F1-score: %.4f' %metrics.f1_score(y_test,y_predict))
        f1.append(round(metrics.f1_score(y_test, y_pred),3))
    acc.append(round(sum(acc) / len(acc),3))
    auc.append(round(sum(auc) / len(auc),3))
    f1.append(round(sum(f1) / len(f1),3))
    return [auc,acc,f1]
content=['/Users/hhy/Desktop/1/test.csv',
         ]
'''
for res in content:
    tmp.append(score(res))'''
index=['auc','acc','f1']
tmp=score('/Users/hhy/Desktop/1/test.csv')
header=[i for i in range(1,11)]
header.append('Mean')
out = pd.DataFrame(tmp,index=index)
out.to_csv('/Users/hhy/Desktop/test/all_feature.csv',encoding='utf-8-sig',header=header)
