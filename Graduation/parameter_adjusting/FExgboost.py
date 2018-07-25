import csv
from sklearn.ensemble import AdaBoostClassifier,GradientBoostingClassifier
from sklearn.grid_search import GridSearchCV
import xgboost as xgb
import numpy as np
data=[]
mark=[]

with open('/Users/hhy/Desktop/test/data.csv','r',encoding='utf-8_sig') as f:
    csv_reader=csv.reader(f)
    for x in csv_reader:
        data.append(list(map(float,x[:-1])))
        mark.append(float(x[-1]))
    data=np.array(data)
#param_test= {'n_estimators':[x for x in range(100,1500,100)]}
#param_test1= {'max_depth':[i for i in range(8,30,2)],'min_child_weight':[j for j in range(1,6,2)]} # 83396
#param_test2 = {'gamma':[i/10.0 for i in range(0,5)]}
#param_test3 = {'learning_rate':[0.09,0.1,0.12,0.14,0.16,0.18,0.2]}
#param_test4 = {'reg_alpha':[1e-5, 1e-2, 0.1, 1, 100]}
param_test5 = {'subsample':[0.6,0.7,0.75,0.8,0.85,0.9,0.95]}
gsearch1= GridSearchCV(estimator = xgb.XGBClassifier(n_estimators=900,learning_rate =0.1,gamma=0.2, max_depth=10,colsample_bytree=0.8,objective= 'binary:logistic', seed=1113),
                       param_grid =param_test5, scoring='roc_auc',cv=5)
gsearch1.fit(data,mark)
print(gsearch1.grid_scores_,gsearch1.best_params_, gsearch1.best_score_)

