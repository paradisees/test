def node_similarity(threshold,method,number):
    import csv
    def cos(v1,v2):
        import math
        def dot_product(v1, v2):
            return sum(a * b for a, b in zip(v1, v2))
        def magnitude(vector):
            return math.sqrt(dot_product(vector, vector))
        def similarity(v1, v2):
            return dot_product(v1, v2) / (magnitude(v1) * magnitude(v2) + .00000000001)
        return round(similarity(v1,v2),3)

    def pearson(vec1, vec2):
        value = range(len(vec1))

        sum_vec1 = sum([vec1[i] for i in value])
        sum_vec2 = sum([vec2[i] for i in value])

        square_sum_vec1 = sum([pow(vec1[i], 2) for i in value])
        square_sum_vec2 = sum([pow(vec2[i], 2) for i in value])

        product = sum([vec1[i] * vec2[i] for i in value])

        numerator = product - (sum_vec1 * sum_vec2 / len(vec1))
        dominator = ((square_sum_vec1 - pow(sum_vec1, 2) / len(vec1)) * (
            square_sum_vec2 - pow(sum_vec2, 2) / len(vec2))) ** 0.5

        if dominator == 0:
            return 0
        result = numerator / (dominator * 1.0)

        return round(result,3)
    herb_name=[]
    with open('/Users/hhy/Desktop/1/node/final/herb_name.csv','r',encoding='utf-8_sig') as f:
        csv_reader=csv.reader(f)
        for x in csv_reader:
            herb_name.append(x[0])
    all_id_feature={}  #包含了所有的对应关系，包括药物，症状，疾病等，name-id
    with open('/Users/hhy/Desktop/1/node/final/id_feature.csv','r',encoding='utf-8_sig') as f:
        csv_reader=csv.reader(f)
        for x in csv_reader:
            all_id_feature[x[0]]=x[1]
    id_feature={}  #只关注药物的id，name-id
    for num in herb_name:
        if num in all_id_feature.keys():   #herb_name中有些药物在网络的三个表里面没出现
            id_feature[num]=all_id_feature[num]
    vector={}  #导入编号对应的向量
    with open('/Users/hhy/Desktop/1/node/final/emb'+str(number)+'.csv','r',encoding='utf-8_sig') as f:
        csv_reader=csv.reader(f)
        for x in csv_reader:
            vector[x[0]]=x[1:]
    '''id_feature 对应为name-向量'''
    for content in id_feature.keys():
        if id_feature[content] in vector.keys():
            id_feature[content]=[float(i) for i in vector[id_feature[content]]]

    '''计算cos和pearson相似性'''
    similarity={}
    for num in herb_name:
        if num in all_id_feature.keys():
            dic = {}
            for content in herb_name:
                if content in all_id_feature.keys() and num!=content:
                    if method == 'cos':
                        dic[content]=cos(id_feature[num],id_feature[content])
                    if method == 'pearson':
                        dic[content]=pearson(id_feature[num],id_feature[content])
            similarity[num]=dic
    #print(similarity)

    '''规定阈值，得到每味中药相似性大于阈值的其他中药'''
    #threshold=0.8
    threshold_similarity={}
    for name in similarity.keys():
        tmp={}
        for content in similarity[name].keys():
            if abs(similarity[name][content])>=threshold:
                tmp[content]=similarity[name][content]
        if tmp:
            threshold_similarity[name]=tmp
    return threshold_similarity
'''
#看看需不要管abs，可能在阈值0.2以上就不存在，一般不用管abs
threshold_similarity=node_similarity(0.1,'pearson')
#print(threshold_similarity)
i=0
for item in threshold_similarity.keys():
    for num in threshold_similarity[item].values():
        if num < 0:
            i+=1
print(i)'''