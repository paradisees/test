import csv
import pandas as pd
from Graduation.node2vector.n2_similarity import node_similarity
method='cos'
#method='pearson'
#set=[0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9]
#set=[0.25,0.3,0.35,0.4]
set=[0.38,0.42,0.45,0.48,0.5,0.55]
#set=[0.26,0.27,0.28,0.29,0.31,0.32,0.33,0.34]
number=[8, 12, 16, 20, 24, 28,32,64,96,128,160,192,256,288,352,416,480,512]
for threshold in set:
    for num in number:
        similarity=node_similarity(threshold,method,num)
        herb_only=[]
        #label=[]
        feature=[]
        #print(similarity)
        with open('/Users/hhy/Desktop/1/herb_only/test.csv','r',encoding='utf-8_sig') as f:
            csv_reader=csv.reader(f)
            for x in csv_reader:
                herb_only.append(list(map(float,x[:])))
                #label.append(x[-1])
        with open('/Users/hhy/Desktop/1/herb_only/feature.csv','r',encoding='utf-8_sig') as f:
            csv_reader=csv.reader(f)
            for x in csv_reader:
                feature.append(x[0])
        fea_number={}
        with open('/Users/hhy/Desktop/1/herb_only/feature.csv','r',encoding='utf-8_sig') as f:
            csv_reader=csv.reader(f)
            i=0
            for x in csv_reader:
                fea_number[x[0]]=i
                i+=1
        def f():
            #新建一个次数矩阵，记录每个数据上共有多少个数据相加，然后求均值
            number=[[1 for i in range(len(herb_only[0])-1)] for j in range(len(herb_only))]
            for i in range(len(herb_only)):
                for j in range(len(herb_only[0])-1):
                    if herb_only[i][j]==1.0 and feature[j] in similarity.keys():
                        tmp=similarity[feature[j]]
                        for key in tmp.keys():
                            if herb_only[i][fea_number[key]]!=1.0:
                                #一般不用管abs，基本没有数据 if abs(tmp[key])>herb_only[i][fea_number[key]]:
                                if tmp[key]>herb_only[i][fea_number[key]]:
                                    herb_only[i][fea_number[key]]=tmp[key]
                                #herb_only[i][fea_number[key]] += tmp[key]
                                #number[i][fea_number[key]]+=1
            for i in range(len(herb_only)):
                for j in range(len(herb_only[0])-1):
                    herb_only[i][j]/=number[i][j]
            return herb_only
        herb_only=f()

        '''from sklearn import preprocessing
        min_max_scaler = preprocessing.MinMaxScaler()
        herb_only_minmax = min_max_scaler.fit_transform(herb_only)
        data = pd.DataFrame(herb_only_minmax)'''
        data = pd.DataFrame(herb_only)
        data.to_csv('/Users/hhy/Desktop/1/node/final/cmp/'+str(num)+'emb'+str(threshold)+'.csv',encoding='utf-8-sig',header=False,index=False)


