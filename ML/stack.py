from sklearn import datasets
import csv
from sklearn import svm
from sklearn import cross_validation

data=[]
mark=[]

with open('/Users/hhy/Desktop/10.csv','r',encoding='utf-8_sig') as f1:
    csv_reader=csv.reader(f1)
    for x in csv_reader:
        data.append(list(map(int,x[0:-1])))
        mark.append(list(map(int, x[-1])))
#iris = datasets.load_iris()
#X, y = iris.data[:, 1:3], iris.target
X=data
#y=mark
y=[]
[y.extend(i) for i in mark]
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from mlxtend.classifier import StackingClassifier
import numpy as np

X_train, X_test, y_train, y_test = cross_validation.train_test_split(
    X, y, test_size=0.3)

clf1 = KNeighborsClassifier(n_neighbors=1)
clf2 = RandomForestClassifier(random_state=1,n_estimators=400)
clf3 = GaussianNB()
clf4=svm.SVC(kernel='linear', C=1)
lr = LogisticRegression()
sclf = StackingClassifier(classifiers=[clf1, clf2, clf3,clf4],
                          meta_classifier=lr)

print('3-fold cross validation:\n')

for clf, label in zip([clf1, clf2, clf3,clf4, sclf],
                      ['KNN',
                       'Random Forest',
                       'Naive Bayes',
                        'SVM',
                       'StackingClassifier']):
    scores = model_selection.cross_val_score(clf, X_train, y_train,
                                             cv=3, scoring='accuracy')
    new = clf.fit(X_train, y_train)
    print(label,':',new.score(X_test, y_test))
    print("Accuracy: %0.2f (+/- %0.2f) [%s]"
          % (scores.mean(), scores.std(), label))