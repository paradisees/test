import numpy as np
from sklearn.lda import LDA
from sklearn import multiclass
import csv
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn import cross_validation
from sklearn import datasets
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
data=[]
mark=[]
with open('/Users/hhy/Desktop/test1.csv','r',encoding='utf-8_sig') as f:
    csv_reader=csv.reader(f)
    for x in csv_reader:
        data.append(list(map(int,x[0:-1])))
        mark.append(int(x[-1]))
res=[]
for i in range(10):
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(
    data, mark, test_size=0.3, random_state=i)
    #print(X_train.shape)
    #clf = svm.SVC(kernel='linear', C=1).fit(X_train, y_train)
    clf = RandomForestClassifier(n_estimators=400).fit(X_train, y_train)
    print(clf.score(X_test, y_test))
    res.append(clf.score(X_test, y_test))
print("==",sum(res)/len(res))