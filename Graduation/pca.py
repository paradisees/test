import csv
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn import cross_validation,metrics
data=[]
mark=[]
with open('/Users/hhy/Desktop/test1.csv','r',encoding='utf-8_sig') as f:
    csv_reader=csv.reader(f)
    for x in csv_reader:
        data.append(list(map(int,x[0:-1])))
        mark.append(int(x[-1]))
pca=PCA(n_components=10)
newData=pca.fit_transform(data)
print(sum(pca.explained_variance_ratio_))

for i in range(10):
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(
    newData, mark, test_size=0.1, random_state=i)
    clf = RandomForestClassifier(n_estimators= 1200)
    clf.fit(X_train, y_train)
    ''' 
    y_predict=clf.predict(X_test)
    test_auc = metrics.roc_auc_score(y_test, y_predict)  # 验证集上的auc值
    print('AUC:',test_auc)'''
    print('准确率:',clf.score(X_test, y_test))