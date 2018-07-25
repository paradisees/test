from sklearn import cross_validation, metrics
from sklearn.ensemble import RandomForestClassifier
import csv
from sklearn.ensemble import AdaBoostClassifier,GradientBoostingClassifier
from sklearn.grid_search import GridSearchCV

data=[]
mark=[]
with open('/Users/hhy/Desktop/test/data.csv','r',encoding='utf-8_sig') as f:
    csv_reader=csv.reader(f)
    for x in csv_reader:
        data.append(list(map(float,x[0:-1])))
        mark.append(float(x[-1]))

#param_test1= {'n_estimators':[x for x in range(100,1500,100)]} # 83396
#param_test2= {'max_depth':[i for i in range(3,21,2)], 'min_samples_split':[j for j in range(5,60,5)]}
#param_test3= {'min_samples_split':[x for x in range(30,70,2)], 'min_samples_leaf':[y for y in range(1,8,1)]}
#param_test4= {'max_features':[x for x in range(7,16,1)]}
#param_test5 = {'subsample':[0.6,0.7,0.75,0.8,0.85,0.9,0.95]}
param_test6 = {'learning_rate':[0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09]}

gsearch1= GridSearchCV(estimator = GradientBoostingClassifier(loss='exponential',learning_rate=0.04,n_estimators=1000,max_depth=7,min_samples_split=48,min_samples_leaf=6,max_features=9,subsample=0.7,
                                  random_state=1113),
                       param_grid =param_test6, scoring='roc_auc',cv=5)
gsearch1.fit(data,mark)
print(gsearch1.grid_scores_,gsearch1.best_params_, gsearch1.best_score_)


'''
y_pred = gsearch1.predict(data)
y_predprob = gsearch1.predict_proba(data)
print('准确率:',gsearch1.score(data, mark))
print ("AUC Score (Train): %f" % metrics.roc_auc_score(mark, y_predprob))
'''