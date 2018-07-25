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
#X_train, X_test, y_train, y_test = cross_validation.train_test_split(
#data, mark, test_size=0.3, random_state=0)
clf = svm.SVC(kernel='linear', C=1)
scores = cross_validation.cross_val_score(clf, data, mark, cv=10)
print(scores,'=======',scores.mean())


#预测
#predicted = cross_validation.cross_val_predict(clf, iris.data,iris.target, cv=10)
#metrics.accuracy_score(iris.target, predicted)