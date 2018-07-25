from sklearn.model_selection import cross_val_score
from sklearn.ensemble import AdaBoostClassifier,GradientBoostingClassifier
from sklearn import cross_validation
import csv
data=[]
mark=[]
with open('/Users/hhy/Desktop/Data100.csv','r',encoding='utf-8_sig') as f:
    csv_reader=csv.reader(f)
    for x in csv_reader:
        data.append(list(map(float,x[0:-1])))
        mark.append(float(x[-1]))
res=[]
for i in range(10):
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(
        data, mark, test_size=0.1,random_state=i)
    #deviance对数似然损失，exponential指数损失（相当于adaboost）
    clf = GradientBoostingClassifier(loss='deviance',
        #loss='exponential',
        learning_rate=0.05,
        n_estimators=2000,
        subsample=0.8,
        max_features='sqrt',
        max_depth=7,).fit(X_train, y_train)  # 迭代100次
    print(clf.score(X_test, y_test))  # 预测准确率
    res.append(clf.score(X_test, y_test))
print("==",sum(res)/len(res))
''' 
from sklearn import metrics
from sklearn.ensemble import ExtraTreesClassifier
model = ExtraTreesClassifier()
model.fit(X_train, y_train)
# display the relative importance of each attribute
print(model.feature_importances_)
'''
