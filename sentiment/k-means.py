from sklearn.cluster import KMeans
import pandas as pd
filename = '/Users/hhy/desktop/1.csv'
data = pd.read_csv(filename,encoding='gbk')
x = data.iloc[0:,0:].as_matrix()
print(x)
num_clusters = 2
km_cluster = KMeans(n_clusters=num_clusters, max_iter=50000, n_init=40, init='k-means++',n_jobs=-1)

#返回各自文本的所被分配到的类索引
result = km_cluster.fit_predict(x)
print(result)

'''
#print ("Predicting result: ", result)
res=[]
for i in range(len(result)):
    res.append(result[i])
#print(res)

myset = set(res)  #myset是另外一个列表，里面的内容是mylist里面的无重复 项
for item in myset:
  print("the %d has found %d" %(item,res.count(item)))

data1 = pd.DataFrame(res)
data1.to_csv('/Users/hhy/Desktop/1/temp.csv')'''