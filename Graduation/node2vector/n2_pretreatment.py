import csv
import pandas as pd
import numpy as np
#定义original，将id-药物，id-症状等信息合并到一个表中
def list_construct(list,path):
    with open(path,encoding='utf-8-sig') as csvfile:
        rows = csv.reader(csvfile)
        for row in rows:
           list.append([row[0],row[1]])
    return list
def original():
    origianl = []
    #list_construct(origianl, '/Users/hhy/Desktop/1/sym.csv')
    #list_construct(origianl, '/Users/hhy/Desktop/1/fsym.csv')
    #list_construct(origianl, '/Users/hhy/Desktop/1/herb.csv')
    #list_construct(origianl, '/Users/hhy/Desktop/1/fherb.csv')
    list_construct(origianl, '/Users/hhy/Desktop/1/herb_effect.csv')
    list_construct(origianl, '/Users/hhy/Desktop/1/herb_sym.csv')
    list_construct(origianl, '/Users/hhy/Desktop/1/herb_disease.csv')
    list_construct(origianl, '/Users/hhy/Desktop/1/herb_target.csv')
    return origianl
original=original()
#为了后续编号，对id和所有特征进行去重
def remove_duplicate(original):
    inhospital_id = []
    feature = []
    for x in original:
        if x[0] not in inhospital_id:
            inhospital_id.append(x[0])
        if x[1] not in feature:
            feature.append(x[1])
    return inhospital_id,feature
inhospital_id,feature=remove_duplicate(original)
#针对id和特征，都需要重新编号
def corresponding(inhospital_id,feature):
    id_feature={}
    #tmp为id和特征的集合，对他们重新编号
    tmp=inhospital_id+feature
    i = 1
    for num in tmp:
        id_feature[num]=i
        i+=1
    return id_feature
id_feature=corresponding(inhospital_id,feature)

#如果需要人的id，则需要对应，否则构建药物网络不用
'''
with open('/Users/hhy/Desktop/1/node/label.csv','r',encoding='utf-8_sig') as f:
    csv_reader=csv.reader(f)
    with open('/Users/hhy/Desktop/1/node/final/label_new.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        for x in csv_reader:
            x[0]=id_feature[x[0]]
            writer.writerow(x)
'''
def DictCSV(fileName="", dataDict={}):
    with open(fileName, "w",encoding='utf-8-sig') as csvFile:
        csvWriter = csv.writer(csvFile)
        for k,v in dataDict.items():
            csvWriter.writerow([k,v])
        csvFile.close()
DictCSV('/Users/hhy/Desktop/1/node/final/id_feature.csv',dataDict=id_feature)

#处理成node2vec的输入数据，将原始original中的边替换为新编码后的
def data(original,id_feature):
    id = []
    new_id = []
    content = []
    new_content = []
    for x in original:
        id.append(x[0])
        content.append(x[1])
    for num in id:
        new_id.append([id_feature[num]])
    for num in content:
        new_content.append([id_feature[num]])
    matrix = np.hstack((new_id, new_content))
    return matrix
input=data(original,id_feature)
out = pd.DataFrame(input)
out.to_csv('/Users/hhy/Desktop/1/node/final/input.csv',encoding='utf-8-sig',header=False,index=False)
