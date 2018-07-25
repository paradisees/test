import pandas as pd
import statsmodels.api as sm
import pylab as pl
import numpy as np
import copy

filename = 'C:/Users/hhy/Desktop/5.csv'
data = pd.read_csv(filename,encoding='gbk')
#print(df.describe())
train_cols=data.columns[:25]
logit=sm.Logit(data['Direction'],data[train_cols])
result=logit.fit()

combos=copy.deepcopy(data)
predict_cols=combos.columns[:25]
combos['predict'] = result.predict(combos[predict_cols])
total = 0
hit = 0
for value in combos.values:
    # 预测分数 predict, 是数据中的最后一列
    predict = value[-1]
    # 实际录取结果
    admit = int(value[25])

    # 假定预测概率大于0.5则表示预测被录取
    if predict > 0.5:
        total += 1
        # 表示预测命中
        if admit == 1:
            hit += 1
print ('Total: %d, Hit: %d, Precision: %.2f' % (total, hit, 100.0*hit/total))
print (result.summary())