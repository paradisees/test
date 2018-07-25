import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
import csv
from sklearn.metrics import confusion_matrix
from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier
train_data=[]
train_mark=[]
test_data=[]
test_mark=[]
with open('/Users/hhy/Desktop/3/vectortrain500.csv','r',encoding='utf-8_sig') as f1:
    csv_reader=csv.reader(f1)
    for x in csv_reader:
        train_data.append(list(map(int,x[0:-1])))
        train_mark.append(int(x[-1]))
with open('/Users/hhy/Desktop/3/vectortest500.csv','r',encoding='utf-8_sig') as f2:
    csv_reader=csv.reader(f2)
    for x in csv_reader:
        test_data.append(list(map(int,x[0:-1])))
        test_mark.append(int(x[-1]))
    #print(test_data)
gbdt=GradientBoostingClassifier(
    loss='deviance',
    learning_rate=0.05,
    n_estimators=2000,
    subsample=0.8,
    max_features=1,
    max_depth=6,
    verbose=1
)
clf=gbdt.fit(train_data,train_mark)
print(clf.score(test_data, test_mark))


'''
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
import csv
from sklearn.metrics import confusion_matrix
from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier
data=[]
mark=[]
with open('/Users/hhy/Desktop/3/vectortrain.csv','r',encoding='utf-8_sig') as f:
    csv_reader=csv.reader(f)
    for x in csv_reader:
        data.append(list(map(int,x[0:-1])))
        mark.append(int(x[-1]))
gbdt=GradientBoostingClassifier(
    loss='deviance',
    learning_rate=0.05,
    n_estimators=500,
    subsample=0.8,
    max_features=1,
    max_depth=3,
    verbose=1
)

X_train, X_test, y_train, y_test = cross_validation.train_test_split(
data, mark, test_size=0.3)
gbdt.fit(X_train,y_train)
clf=gbdt.fit(X_train,y_train)
print(clf.score(X_test, y_test))


'''