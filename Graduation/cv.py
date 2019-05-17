import numpy as np
from sklearn.lda import LDA
from sklearn import multiclass
import csv
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.ensemble import RandomForestClassifier

data=[]
mark=[]
with open('/Users/hhy/Desktop/test/data.csv','r',encoding='utf-8_sig') as f:
    csv_reader=csv.reader(f)
    for x in csv_reader:
        data.append(list(map(int,x[0:-1])))
        mark.append(int(x[-1]))

result=[]

for i in range(10):

    head=int((len(mark)/10)*i)
    tail=int((len(mark)/10)*(i+1))


    #x_train = np.array(data[0:head]+data[tail:])
    #y_train = np.array(mark[0:head]+mark[tail:])
    x_train = np.vstack((data[0:head], data[tail:]))
    y_train = np.vstack((mark[0:head], mark[tail:]))
    x_test = np.array(data[head:tail])
    y_test = np.array(mark[head:tail])
    try:
      #clf=LDA()
      #clf.fit(x_train, y_train)
      clf = RandomForestClassifier(n_estimators=400).fit(x_train, y_train)
      r=clf.score(x_test, y_test)
      print(r)
      result.append(r)
    except:
        continue


print('average:%f' %(sum(result)/len(result)))
