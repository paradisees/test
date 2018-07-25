import csv
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.grid_search import GridSearchCV
from sklearn import svm
import numpy as np
data=[]
mark=[]
with open('/Users/hhy/Desktop/test/data.csv','r',encoding='utf-8_sig') as f:
    csv_reader=csv.reader(f)
    for x in csv_reader:
        data.append(list(map(float,x[0:-1])))
        mark.append(float(x[-1]))

param_test1= {'C':[x for x in np.arange(2,10,0.2)]}
parameters = {'kernel': ('linear', 'sigmoid'), 'C': [2,2.1,2.2],'gamma':[0.00001,0.0001,0.0003,0.0005,0.001,0.002]}

#cv = cross_validation.ShuffleSplit(len(data), n_iter=5, test_size=0.1, random_state=i)

gsearch1= GridSearchCV(estimator = svm.SVC(random_state=1113),
                       param_grid =parameters, scoring='roc_auc',cv=5)
gsearch1.fit(data,mark)
print(gsearch1.grid_scores_,gsearch1.best_params_, gsearch1.best_score_)


'''
y_pred = gsearch1.predict(data)
y_predprob = gsearch1.predict_proba(data)
print('准确率:',gsearch1.score(data, mark))
print ("AUC Score (Train): %f" % metrics.roc_auc_score(mark, y_predprob))
'''