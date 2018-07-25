from sklearn import cross_validation,metrics
from sklearn import svm
from sklearn.linear_model.logistic import LogisticRegression

def forward(data,mark):
    res=[]
    for i in range(100):
        X_train, X_test, y_train, y_test = cross_validation.train_test_split(
        data, mark, test_size=0.05, random_state=i)
        clf = svm.SVC(kernel='linear', C=2)   #0.7786  7906
        #clf = RandomForestClassifier(n_estimators= 800)
        #clf = LogisticRegression()   #0.7864
        clf.fit(X_train, y_train)
        y_predict = clf.predict(X_test)
        test_auc = metrics.roc_auc_score(y_test, y_predict)  # 验证集上的auc值
        #print('AUC:', test_auc)
        res.append(test_auc)
    return (sum(res)/len(res))


#检测forward算法有木有毛病
def test_forward():
    import csv
    names,score=[],[]
    i=0
    with open('/Users/hhy/Desktop/fea_importance.csv', 'r', encoding='utf-8_sig') as f1:
        csv_reader = csv.reader(f1)
        for x in csv_reader:
            if i==0:
                i+=1
                continue
            else:
                names.append(x[0])
                score.append(x[-1])
    fea = {}
    for j in range(len(names)):
        fea[names[j]] = score[j]
    new = sorted(fea.items(), key=lambda x: (x[1],x[0]), reverse=True)
    return new
