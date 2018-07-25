from sklearn.cluster import KMeans,DBSCAN
from sklearn.externals import joblib
import pandas as pd
import csv
#kMeans
filename = '/Users/hhy/desktop/3/vector1.csv'
data = pd.read_csv(filename,encoding='gbk')
x = data.iloc[0:,0:].as_matrix()
print(x[0],len(x))

#建模
num_clusters = 50
km_cluster = KMeans(n_clusters=num_clusters, max_iter=50000, n_init=40, init='k-means++',n_jobs=-1)
#返回各自文本的所被分配到的类索引
#print(result)
#joblib.dump(km_cluster, '/Users/hhy/Desktop/3/doc_cluster.pkl')


#km_cluster=joblib.load('/Users/hhy/Desktop/3/doc_cluster.pkl')
result = km_cluster.fit_predict(x)
print(len(result))

data = pd.DataFrame(result)
data.to_csv('/Users/hhy/Desktop/cat50.csv')



