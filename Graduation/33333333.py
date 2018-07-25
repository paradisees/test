from sklearn import datasets
import csv
from sklearn import svm
from sklearn import cross_validation
import numpy as np
data=[]
mark=[]

with open('/Users/hhy/Desktop/test/data1.csv','r',encoding='utf-8_sig') as f1:
    csv_reader=csv.reader(f1)
    for x in csv_reader:
        data.append(list(map(float,x[0:-1])))
        mark.append(list(map(float,x[-1])))
X=np.array(data)
#y=mark
y=[]
[y.extend(i) for i in mark]
y=np.array(y)
print(X,y)
'''
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from mlxtend.classifier import StackingClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import ExtraTreesClassifier
import xgboost as xgb
from sklearn.naive_bayes import GaussianNB

import numpy as np
#clf1 = LogisticRegression(C=4.7)  # 0.79027075505
clf1 = svm.SVC(kernel='linear', C=2)
#clf7 = ExtraTreesClassifier()
#clf8 = xgb.XGBClassifier()
#clf9 = GaussianNB()
lr = LogisticRegression(C=4.7)
sclf = StackingClassifier(classifiers=[clf1],
                          meta_classifier=lr)

res=[]
for i in range(3):
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(
        X, y, test_size=0.1,random_state=i)

    for clf, label in zip([clf1,sclf],
                          ['SVM',
                           'StackingClassifier']):
        clf.fit(X_train, y_train)
        print('准确率:',clf.score(X_test, y_test))
        res.append(clf.score(X_test, y_test))
print(sum(res)/len(res))
'''