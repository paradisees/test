import numpy as np
from sklearn.lda import LDA
from sklearn import cross_validation,metrics
from sklearn.linear_model import (LinearRegression, Ridge,Lasso, RandomizedLasso)
from sklearn import svm
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.linear_model import RidgeCV,LassoCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn import svm, datasets
from sklearn.cross_validation import StratifiedKFold
import csv
from sklearn.ensemble import AdaBoostClassifier,GradientBoostingClassifier

def roc_plot():
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, auc
    from scipy import interp
    mean_tpr = 0.0
    mean_fpr = np.linspace(0, 1, 100)
    all_tpr = []
    #style=['b^','gs','ro','cp','mD']
    for i in range(5):
        X_train, X_test, y_train, y_test = cross_validation.train_test_split(
            data, mark, test_size=0.05, random_state=i)
        clf = LogisticRegression(C=4.8, random_state=1113)
        #clf = svm.SVC(kernel='linear', C=3, probability=True, gamma=0.001, random_state=1113)
        name='Logistic cv'+str(i)
        # 通过训练数据，使用svm线性核建立模型，并对测试集进行测试，求出预测得分
        probas_ = clf.fit(X_train, y_train).predict_proba(X_test)
        fpr, tpr, thresholds = roc_curve(y_test, probas_[:, 1])
        mean_tpr += interp(mean_fpr, fpr, tpr)  # 对mean_tpr在mean_fpr处进行插值，通过scipy包调用interp()函数
        mean_tpr[0] = 0.0  # 初始处为0
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=1, label='%s ROC (area = %0.3f)' % (name,roc_auc))
    # 画对角线
    plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')
    mean_tpr /= 5  # 在mean_fpr100个点，每个点处插值插值多次取平均
    mean_tpr[-1] = 1.0  # 坐标最后一个点为（1,1）
    mean_auc = auc(mean_fpr, mean_tpr)  # 计算平均AUC值
    plt.plot(mean_fpr, mean_tpr, 'k--',
             label='Mean ROC (area = %0.3f)' % mean_auc, lw=2)

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
roc_plot()
