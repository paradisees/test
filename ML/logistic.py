import pandas as pd
#参数初始化
filename = '/Users/hhy/desktop/testtest.csv'
data = pd.read_csv(filename,encoding='gbk')
x = data.iloc[:,:2670].as_matrix()
y = data.iloc[:,2670].as_matrix()

from sklearn.linear_model import LogisticRegression as LR
from sklearn.linear_model import RandomizedLogisticRegression as RLR
rlr = RLR() #建立随机逻辑回归模型，筛选变量
rlr.fit(x, y) #训练模型

rlr.get_support() #获取特征筛选结果，也可以通过.scores_方法获取各个特征的分数
print(u'通过随机逻辑回归模型筛选特征结束。')
print(u'有效特征为：%s' % ','.join(data.columns[rlr.get_support()]))
x = data[data.columns[rlr.get_support()]].as_matrix() #筛选好特征
#print(x)
#print(y)
lr = LR() #建立逻辑回归模型
lr.fit(x, y) #用筛选后的特征数据来训练模型

print(u'逻辑回归模型训练结束。')
print(u'模型的平均正确率为：%s' % lr.score(x, y)) #给出模型的平均正确率
