import numpy as np
from sklearn.lda import LDA
from sklearn import cross_validation,metrics
from sklearn.linear_model import (LinearRegression, Ridge,Lasso, RandomizedLasso)
from sklearn import svm
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.linear_model import RidgeCV,LassoCV
from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier
from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.cross_validation import StratifiedKFold
import csv
from sklearn.ensemble import AdaBoostClassifier,GradientBoostingClassifier
import xgboost as xgb
def roc_plot(clfs):
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, auc
    from scipy import interp
    style=['r-','g--','b-']
    a = 0
    lw=[1,1,2]
    for clf,name in clfs:
        mean_tpr = 0.0
        mean_fpr = np.linspace(0, 1, 100)
        all_tpr = []
        for i in range(5):
            X_train, X_test, y_train, y_test = cross_validation.train_test_split(
                data, mark, test_size=0.05, random_state=i)
            probas_ = clf.fit(X_train, y_train).predict_proba(X_test)
            fpr, tpr, thresholds = roc_curve(y_test, probas_[:, 1])
            mean_tpr += interp(mean_fpr, fpr, tpr)  # 对mean_tpr在mean_fpr处进行插值，通过scipy包调用interp()函数
            mean_tpr[0] = 0.0  # 初始处为0
            roc_auc = auc(fpr, tpr)
            #plt.plot(fpr, tpr, lw=1, label='%s ROC (area = %0.3f)' % (name,roc_auc))
        # 画对角线
        mean_tpr /= 5  # 在mean_fpr100个点，每个点处插值插值多次取平均
        mean_tpr[-1] = 1.0  # 坐标最后一个点为（1,1）
        mean_auc = auc(mean_fpr, mean_tpr)  # 计算平均AUC值
        plt.plot(mean_fpr, mean_tpr,style[a],
                 label='%s Mean ROC (area = %0.3f)' % (name,mean_auc), lw=1)
        a+=1
    plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()
data=[]
mark=[]
with open('/Users/hhy/Desktop/test/data.csv','r',encoding='utf-8_sig') as f:
    csv_reader=csv.reader(f)
    for x in csv_reader:
        data.append(list(map(float,x[0:-1])))
        mark.append(int(x[-1]))
data=np.array(data)
mark=np.array(mark)
res=[]
train=[]
#clf1 = LogisticRegression(C=4.8,random_state=1113)  #0.8812
#clf2 = svm.SVC(kernel='linear', C=3,probability=True,gamma=0.001,random_state=1113) #0.8824
clf3 = GradientBoostingClassifier(learning_rate=0.03,random_state=1113,n_estimators=1100,min_samples_split=28,min_samples_leaf=4,max_depth=5,max_features=9,subsample=0.8)  #0.8673
clf4 = GradientBoostingClassifier(loss='exponential', learning_rate=0.04, n_estimators=1000, max_depth=7,min_samples_split=48, min_samples_leaf=6, max_features=9, subsample=0.7,random_state=1113)   #adaboost  0.8612
clf5 = xgb.XGBClassifier(n_estimators=900,learning_rate =0.1,gamma=0.2, subsample=0.8,max_depth=10,colsample_bytree=0.8,objective= 'binary:logistic', seed=1113)
#clf6 = RandomForestClassifier(random_state=1113,n_estimators=900,max_depth=24,max_features=8,min_samples_split=2)  #0.8443
#clf7 = ExtraTreesClassifier(random_state=1113, n_estimators=1800, max_depth=28, max_features=15) #0.8538
clfs=zip([clf5,clf4,clf3],['Xgboost','Adaboost','GBDT'])
roc_plot(clfs)
