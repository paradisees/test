import numpy as np
from sklearn.lda import LDA
from sklearn import cross_validation,metrics
from sklearn.linear_model import (LinearRegression, Ridge,Lasso, RandomizedLasso)
from sklearn import svm
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.linear_model import RidgeCV,LassoCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from scipy import interp
import matplotlib.pyplot as plt
from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.cross_validation import StratifiedKFold
import csv
data=[]
mark=[]
with open('/Users/hhy/Desktop/test/data.csv','r',encoding='utf-8_sig') as f:
    csv_reader=csv.reader(f)
    for x in csv_reader:
        data.append(list(map(float,x[0:-1])))
        mark.append(int(x[-1]))
res=[]
train=[]
X_train, X_test, y_train, y_test = cross_validation.train_test_split(
data, mark, test_size=0.1, random_state=0)
#clf = LogisticRegression(C=4.8,random_state=1113)
clf = svm.SVC(kernel='linear', C=2,probability=True)
clf.fit(X_train, y_train)
y_predict = clf.predict_proba(X_test)[:,1]
test_auc = metrics.roc_auc_score(y_test, y_predict)  # 验证集上的auc值
print('AUC:', test_auc)

data=np.array(data)
mark=np.array(mark)
cv = StratifiedKFold(mark, n_folds=5)
print(cv)
mean_tpr = 0.0
mean_fpr = np.linspace(0, 1, 100)
all_tpr = []

for i, (train, test) in enumerate(cv):
    # 通过训练数据，使用svm线性核建立模型，并对测试集进行测试，求出预测得分
    probas_ = clf.fit(data[train], mark[train]).predict_proba(data[test])
    #    print set(y[train])                     #set([0,1]) 即label有两个类别
    #    print len(X[train]),len(X[test])        #训练集有84个，测试集有16个
    #    print "++",probas_                      #predict_proba()函数输出的是测试集在lael各类别上的置信度，
    #    #在哪个类别上的置信度高，则分为哪类
    # Compute ROC curve and area the curve
    # 通过roc_curve()函数，求出fpr和tpr，以及阈值
    fpr, tpr, thresholds = roc_curve(mark[test], probas_[:, 1])
    mean_tpr += interp(mean_fpr, fpr, tpr)  # 对mean_tpr在mean_fpr处进行插值，通过scipy包调用interp()函数
    mean_tpr[0] = 0.0  # 初始处为0
    roc_auc = auc(fpr, tpr)
    # 画图，只需要plt.plot(fpr,tpr),变量roc_auc只是记录auc的值，通过auc()函数能计算出来
    plt.plot(fpr, tpr, lw=1, label='ROC fold %d (area = %0.2f)' % (i, roc_auc))

# 画对角线
plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')

mean_tpr /= len(cv)  # 在mean_fpr100个点，每个点处插值插值多次取平均
mean_tpr[-1] = 1.0  # 坐标最后一个点为（1,1）
mean_auc = auc(mean_fpr, mean_tpr)  # 计算平均AUC值
# 画平均ROC曲线
# print mean_fpr,len(mean_fpr)
# print mean_tpr
plt.plot(mean_fpr, mean_tpr, 'k--',
         label='Mean ROC (area = %0.2f)' % mean_auc, lw=2)

plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()