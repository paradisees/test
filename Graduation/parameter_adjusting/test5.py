from sklearn.linear_model import (LinearRegression, Ridge,Lasso, RandomizedLasso)
from sklearn.feature_selection import RFE, f_classif
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import RidgeCV,LassoCV
from sklearn import cross_validation,metrics
import numpy as np
from minepy import MINE
import csv
import pandas as pd
data=[]
key=[0.684,0.753,0.707,0.754,0.686,0.747,0.707,1]
'''Extra
GBDT
Lasso
MIC
RF
RFE
Ridge
stability
Value'''
out=[]
with open('/Users/hhy/Desktop/csv.csv','r',encoding='utf-8_sig') as f:
    csv_reader=csv.reader(f)
    for x in csv_reader:
        data.append(list(map(float,x[1:])))
for num in data:
    tmp=0
    for i in range(len(num)):
        tmp+=key[i]*num[i]
    out.append(tmp/8)
print(out)