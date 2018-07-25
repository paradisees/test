import csv
from sklearn.grid_search import GridSearchCV
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
data=[]
mark=[]

with open('/Users/hhy/Desktop/Data100.csv','r',encoding='utf-8_sig') as f:
    csv_reader=csv.reader(f)
    for x in csv_reader:
        data.append(list(map(float,x[:-1])))
        mark.append(float(x[-1]))
    data=np.array(data)
param_test1= {'n_neighbors':[i for i in range(1,30,1)]} # 83396

gsearch1= GridSearchCV(estimator = KNeighborsClassifier(),
                       param_grid =param_test1, scoring='roc_auc',cv=10)
gsearch1.fit(data,mark)
print(gsearch1.grid_scores_,gsearch1.best_params_, gsearch1.best_score_)

