from sklearn import  metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import csv
data=[]
mark=[]
with open('/Users/hhy/Desktop/Data100.csv','r',encoding='utf-8_sig') as f:
    csv_reader=csv.reader(f)
    for x in csv_reader:
        data.append(list(map(float,x[0:-1])))
        mark.append(float(x[-1]))
def png(neibors_settings):
    plt.plot(neibors_settings,training_acc,label='training acc')
    plt.plot(neibors_settings,test_acc,label='test acc')
    plt.ylabel("acc")
    plt.xlabel("n_neighbors")
    plt.legend()
    plt.savefig('knn')
for i in range(100):
    X_train, X_test, y_train, y_test = train_test_split(
        data, mark, test_size=0.1,random_state=i)
    training_acc,test_acc=[],[]
    #neibors_settings=range(1,30)
    #for n_neighbors in neibors_settings:
    knn=KNeighborsClassifier(n_neighbors=13)
    knn.fit(X_train,y_train)
    training_acc.append(knn.score(X_train,y_train))
    test_acc.append(knn.score(X_test,y_test))
print(sum(training_acc)/len(training_acc),sum(test_acc)/len(test_acc))

