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
clf1 = svm.SVC(kernel='linear', C=2)
clf2 = RandomForestClassifier(random_state=1113,n_jobs=4,max_depth=17,n_estimators=1000,max_features=7)
clf3 = ExtraTreesClassifier(n_estimators=800, n_jobs=-1, criterion='entropy')
clf4 = GradientBoostingClassifier(learning_rate=0.03,n_estimators=800,min_samples_leaf=2,min_samples_split=36,max_depth=7,max_features=13, subsample=0.9,random_state=1113)
clf6 = GradientBoostingClassifier(loss='exponential',n_estimators=400,max_depth=5,min_samples_split=22,learning_rate=0.07,
                                  max_features='sqrt', subsample=0.75,random_state=1113)
clf7 = xgb.XGBClassifier(learning_rate =0.1,n_estimators=400,max_depth=16,gamma=0,subsample=0.8,colsample_bytree=0.8,objective= 'binary:logistic',nthread=4,seed=1113)

#clf7 = ExtraTreesClassifier()
#clf8 = xgb.XGBClassifier()
#clf9 = GaussianNB()
lr = LogisticRegression()
sclf = StackingClassifier(classifiers=[clf1, clf2, clf3,clf4,clf6,clf7],
                          meta_classifier=lr)

res=[]
for i in range(3):
    #X_train, X_test, y_train, y_test = cross_validation.train_test_split(
    #    X, y, test_size=0.1,random_state=i)

    for clf, label in zip([clf1, clf2, clf3,clf4,clf6,clf7, sclf],
                          ['SVM',
                           'Random Forest',
                           'extra',
                           'GBDT',
                           'adaboost',
                           'xgboost',
                           'StackingClassifier']):
        cv = cross_validation.ShuffleSplit(len(data), n_iter=5, test_size=0.1, random_state=i)
        scores = model_selection.cross_val_score(clf, X, y,cv=cv, scoring='accuracy')
        #new = clf.fit(X_train, y_train)
        #print(label,':',new.score(X_test, y_test))
        print("Accuracy: %0.4f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))
        res.append(scores.mean())
print(sum(res)/len(res))
'''Accuracy: 0.7865 (+/- 0.03) [SVM]
Accuracy: 0.7560 (+/- 0.03) [Random Forest]
Accuracy: 0.7666 (+/- 0.04) [extra]
Accuracy: 0.7689 (+/- 0.04) [GBDT]
Accuracy: 0.7766 (+/- 0.04) [adaboost]
Accuracy: 0.7830 (+/- 0.03) [StackingClassifier]
'''