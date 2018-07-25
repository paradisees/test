from sklearn import cross_validation, metrics
from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier
import csv
from sklearn.grid_search import GridSearchCV
from sklearn.linear_model.logistic import LogisticRegression
import numpy as np
from sklearn import svm
data=[]
mark=[]
with open('/Users/hhy/Desktop/test/data.csv','r',encoding='utf-8_sig') as f:
    csv_reader=csv.reader(f)
    for x in csv_reader:
        data.append(list(map(float,x[0:-1])))
        mark.append(float(x[-1]))
#parameters = {'kernel': ('linear', 'sigmoid'), 'C': [1, 2],'gamma': [0.001, 0.002, 0.004, 0.006, 0.01, 0.02, 0.04, 0.1]}
#parameters = {'kernel': ('linear', 'sigmoid'), 'C': [1,2],'gamma':[0.001,0.002,0.004,0.006,0.01,0.02,0.04,0.1]}
#param_test1= {'C':[i for i in np.arange(0.5,5,0.1)]}
#param_test1= {'n_estimators':[x for x in range(600,2000,100)]}
#param_test2= {'max_depth':[i for i in range(18,30,2)], 'min_samples_split':[j for j in range(2,6,1)]}
#param_test3= {'min_samples_split':[x for x in range(2,10,1)], 'min_samples_leaf':[y for y in range(2,40,3)]}
param_test4= {'max_features':[x for x in range(7,16,1)]}
#cv = cross_validation.ShuffleSplit(len(data), n_iter=5, test_size=0.1, random_state=i)

gsearch1= GridSearchCV(estimator = RandomForestClassifier(random_state=1113,n_estimators=900,max_depth=24,max_features=8,min_samples_split=2),
                       param_grid =param_test4, scoring='roc_auc',cv=5)
gsearch1.fit(data,mark)
print(gsearch1.grid_scores_,gsearch1.best_params_, gsearch1.best_score_)

#,n_estimators=900,max_depth=15,min_samples_split=15,min_samples_leaf=10

'''random_state=1113,n_jobs=4,max_depth=17,n_estimators=1000,max_features=10'''

'''n_estimators=2100,'max_depth': 19, 'min_samples_split': 5'''