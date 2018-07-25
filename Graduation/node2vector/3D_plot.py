import numpy as np
import matplotlib.pyplot as plt
import csv
from sklearn import cross_validation,metrics
from sklearn.linear_model.logistic import LogisticRegression
from mpl_toolkits.mplot3d import Axes3D

set=[0.25,0.3,0.35,0.4]
number=[8, 12, 16, 20, 24, 28,32,64,96,128,160,192,256,288,352,416,480,512]
x_cos,y_cos,z_cos=[],[],[]
x_pearson,y_pearson,z_pearson=[],[],[]
for threshold in set:
    for num in number:
        data = []
        mark = []
        with open('/Users/hhy/Desktop/1/node/final/cmp/meancos'+str(num)+'emb'+str(threshold)+'.csv', 'r',encoding='utf-8_sig') as f:
            csv_reader=csv.reader(f)
            for x in csv_reader:
                data.append(list(map(float,x[0:-1])))
                mark.append(float(x[-1]))
        tmp_acc = []
        tmp_auc = []
        x_cos.append(float(num))
        y_cos.append(float(threshold))
        for i in range(10):
            X_train, X_test, y_train, y_test = cross_validation.train_test_split(
            data, mark, test_size=0.05, random_state=i)
            clf = LogisticRegression(C=4.8,random_state=1113)
            clf.fit(X_train, y_train)
            tmp_acc.append(clf.score(X_test, y_test))
            y_predict = clf.predict_proba(X_test)[:, 1]
            tmp_auc.append(metrics.roc_auc_score(y_test, y_predict))  # 验证集上的auc值
        z_cos.append(round(sum(tmp_acc)/len(tmp_acc),3))
for threshold in set:
    for num in number:
        data = []
        mark = []
        with open('/Users/hhy/Desktop/1/node/final/cmp/meanpearson'+str(num)+'emb'+str(threshold)+'.csv', 'r',encoding='utf-8_sig') as f:
            csv_reader=csv.reader(f)
            for x in csv_reader:
                data.append(list(map(float,x[0:-1])))
                mark.append(float(x[-1]))
        tmp_acc = []
        tmp_auc = []
        x_pearson.append(float(num))
        y_pearson.append(float(threshold))
        for i in range(10):
            X_train, X_test, y_train, y_test = cross_validation.train_test_split(
            data, mark, test_size=0.05, random_state=i)
            clf = LogisticRegression(C=4.8,random_state=1113)
            clf.fit(X_train, y_train)
            tmp_acc.append(clf.score(X_test, y_test))
            y_predict = clf.predict_proba(X_test)[:, 1]
            tmp_auc.append(metrics.roc_auc_score(y_test, y_predict))  # 验证集上的auc值
        z_pearson.append(round(sum(tmp_acc)/len(tmp_acc),3))
ax = plt.subplot(111, projection='3d')  # 创建一个三维的绘图工程
#  将数据点分成三部分画，在颜色上有区分度
ax.scatter(x_cos,y_cos,z_cos, c='b')  # 绘制数据点
ax.scatter(x_pearson,y_pearson,z_pearson, c='r',marker='^')  # 绘制数据点
plt.legend(['cos', 'pearson'], loc = 'upper left')
ax.set_zlabel('ACC')  # 坐标轴
ax.set_ylabel('Threshold')
ax.set_xlabel('Dimension')
plt.show()