from mlxtend.classifier import StackingClassifier
from mlxtend.feature_selection import ColumnSelector
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.externals import joblib
from sklearn.ensemble import AdaBoostClassifier,GradientBoostingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn import svm
from sklearn import cross_validation
import numpy as np
import csv
data=[]
mark=[]
with open('/Users/hhy/Desktop/test/Data.csv','r',encoding='utf-8_sig') as f:
    csv_reader=csv.reader(f)
    for x in csv_reader:
        data.append(list(map(float,x[0:-1])))
        mark.append(float(x[-1]))
    data=np.array(data)

from Graduation.fea_engineer import score,n_feature,modify_data
stacking=['lr','svm','rf','extra','gbdt','adaboost']
stacking_dic={}
out={}
for flag in stacking:
    feature_count = [i for i in range(50, 151, 5)]
    id, tmp = 0, 0  # tmp为特征个数
    for content in feature_count:
        feature = n_feature(content)
        new_data = modify_data(feature)
        Score = score(new_data,flag)
        if Score > id:
            id = Score
            tmp = content
    print(flag)
    #print('%s得分较高：' % flag,[tmp, id])
    stacking_dic[flag]=tmp
    out[flag]=(tmp,id)
print(stacking_dic)
print(out)
def correspond(flag):
    if flag=='lr':
        return LogisticRegression(random_state=1113)
    elif flag=='svm':
        return svm.SVC(random_state=1113)
    elif flag=='rf':
        return RandomForestClassifier(random_state=1113)
    elif flag=='extra':
        return ExtraTreesClassifier(random_state=1113)
    elif flag=='gbdt':
        return GradientBoostingClassifier(random_state=1113)
    elif flag=='adaboost':
        return GradientBoostingClassifier(random_state=1113,loss='exponential')
pipe=[]
i=1
for item in stacking_dic.keys():
    locals()['pipe'+str(i)] = make_pipeline(ColumnSelector(cols=[j for j in range(stacking_dic[item])]),
                      correspond(item))
    pipe.append(locals()['pipe'+str(i)])
    i+=1
sclf = StackingClassifier(classifiers=pipe,
                          meta_classifier=LogisticRegression())
joblib.dump(sclf,'sclf.model')
sclf.fit(data, mark)
print('准确率:',sclf.score(data, mark))

'''== 0.732394366197'''