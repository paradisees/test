from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import StratifiedKFold
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.datasets.samples_generator import make_blobs
import csv
from sklearn import svm
from sklearn import cross_validation
from sklearn.linear_model.logistic import LogisticRegression
def roc_plot(clfs,X_train, y_train,X_test,y_test):
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, auc
    from scipy import interp
    for clf,name in clfs:
        mean_tpr = 0.0
        mean_fpr = np.linspace(0, 1, 100)
        all_tpr = []
        # 通过训练数据，使用svm线性核建立模型，并对测试集进行测试，求出预测得分
        probas_ = clf.fit(X_train, y_train).predict_proba(X_test)
        fpr, tpr, thresholds = roc_curve(y_test, probas_[:, 1])
        mean_tpr += interp(mean_fpr, fpr, tpr)  # 对mean_tpr在mean_fpr处进行插值，通过scipy包调用interp()函数
        mean_tpr[0] = 0.0  # 初始处为0
        roc_auc = auc(fpr, tpr)
        # 画图，只需要plt.plot(fpr,tpr),变量roc_auc只是记录auc的值，通过auc()函数能计算出来
        plt.plot(fpr, tpr, lw=1, label='%s ROC (area = %0.2f)' % (name,roc_auc))
    # 画对角线
    plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')
    mean_tpr /= 10  # 在mean_fpr100个点，每个点处插值插值多次取平均
    mean_tpr[-1] = 1.0  # 坐标最后一个点为（1,1）
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()
data=[]
mark=[]
with open('/Users/hhy/Desktop/test/data1.csv','r',encoding='utf-8_sig') as f1:
    csv_reader=csv.reader(f1)
    for x in csv_reader:
        data.append(list(map(float,x[0:-1])))
        mark.append(list(map(float,x[-1])))
X=np.array(data)
#y=mark
target=[]
[target.extend(i) for i in mark]
target=np.array(target)
'''模型融合中使用到的各个单模型'''
clfs = [LogisticRegression(C=4.7),
        svm.SVC(kernel='linear', C=2,probability=True),
        RandomForestClassifier(n_estimators=5, n_jobs=-1, criterion='entropy',random_state=2018),
        ExtraTreesClassifier(n_estimators=5, n_jobs=-1, criterion='gini',random_state=2018),
        ExtraTreesClassifier(n_estimators=5, n_jobs=-1, criterion='entropy',random_state=2018),
        GradientBoostingClassifier(learning_rate=0.05, subsample=0.5, max_depth=6, n_estimators=5,random_state=2018)
         ]
'''切分一部分数据作为测试集'''
#X-y（训练集）   X_predict-y_predict（测试集）
X, X_predict, y, y_predict = train_test_split(X, target, test_size=0.1, random_state=0)
#第二层模型的训练和测试集
dataset_blend_train = np.zeros((X.shape[0], len(clfs)))  #1265*2
dataset_blend_test = np.zeros((X_predict.shape[0], len(clfs))) #141*2

'''10折stacking'''
n_folds = 10
#切分训练集标签长度为十份，做交叉验证
skf = list(StratifiedKFold(y, n_folds))  #总长度为10，每部分分为训练和测试的下标号
for j, clf in enumerate(clfs):
    '''依次训练各个单模型'''
    dataset_blend_test_j = np.zeros((X_predict.shape[0], len(skf))) #141*10
    for i, (train, test) in enumerate(skf):
        '''使用第i个部分作为预测，剩余的部分来训练模型，获得其预测的输出作为第i部分的新特征。'''
        X_train, y_train, X_test, y_test = X[train], y[train], X[test], y[test]
        clf.fit(X_train, y_train)
        y_submission = clf.predict_proba(X_test)[:, 1]
        dataset_blend_train[test, j] = y_submission
        #print(clf.predict_proba(X_predict)[:, 1])
        dataset_blend_test_j[:, i] = clf.predict_proba(X_predict)[:, 1]
    #roc_plot(X_train, y_train)
    '''对于测试集，直接用这k个模型的预测值均值作为新的特征。'''
    dataset_blend_test[:, j] = dataset_blend_test_j.mean(1)
    print("val auc Score: %f" % roc_auc_score(y_predict, dataset_blend_test[:, j]))
    #print(y_predict,dataset_blend_test[:, j])

clf = LogisticRegression(C=4.7)
#clf = svm.SVC(kernel='linear', C=2)
clf.fit(dataset_blend_train, y)
y_submission = clf.predict_proba(dataset_blend_test)[:, 1]
print("blend result val auc Score: %f" % (roc_auc_score(y_predict, y_submission)))

#roc_plot(dataset_blend_train, y)