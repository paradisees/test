import nltk
import csv
import numpy as np
import pandas as pd

#提取NLTK处理后的词，再进行主语提取
def fea(text):
    with open(text) as f:
        rows = csv.reader(f)
        res=[]
        s=' '
        for row in rows:
            res1=[]
            for i in range(len(row)):
                if row[i]!='' and row[i]!='.':
                    res1.append(row[i])
            res.append(s.join(res1))
        #fea = list(set(res))
        return res
m=fea('/Users/hhy/Desktop/1/eng3.csv')
#print(m)

#提取主语
def noun(text):
    sens=nltk.sent_tokenize(text)
    words=[]
    for sent in sens:
        words.append(nltk.word_tokenize(sent))
    tags=[]
    for tokens in words:
        tags.append(nltk.pos_tag(tokens))
    res=[]
    for tag in tags:
        ners=nltk.ne_chunk(tag)
        res.append(ners)
    n = len(res[0])
    res1 = []
    for i in range(n):
        if res[0][i][1] == 'NN' or res[0][i][1] == 'NNS' or res[0][i][1] == 'NNP':
            res1.append(res[0][i][0])
    return res1
#noun=noun('x0fs x07')
#print(noun)

#strs:得到每条评论的主语集合
def collection():
    strs=[]
    for res in m:
        strs.append(noun(res))
    return strs

strs=collection()
with open('/Users/hhy/Desktop/1/comments.csv', 'w', encoding='gb18030', newline='') as f:
    writer = csv.writer(f)
    for i in range(len(strs)):
        writer.writerow(strs[i])
#print(strs)

#构造评论的字典，根据序号取索引值
def dic():
    dic={}
    for i in range(5000):
        dic[str(i)] = []
    i = 0
    # 按序号在字典中添加词
    for str1 in strs:
        if i <= 5000:
            for j in range(len(str1)):
                if str1[j] != '':
                    dic[str(i)].append(str(str1[j]))
            i += 1
    return dic
dic=dic()
#print(dic)

#得到评论主语所有单词的集合
def getNouns(strs):
    nouns=[]
    for str in strs:
        for i in range(len(str)):
            if str[i] not in nouns:
                nouns.append(str[i])
    return nouns
nouns=getNouns(strs)
with open('/Users/hhy/Desktop/1/comment1.csv', 'w', encoding='gb18030', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(nouns)
#print(nouns)
print(len(nouns))

#构建矩阵
def matrix():
    b = np.zeros((5000, 4360))
    a = b.astype(int)
    for i in range(len(nouns)):
        for j in range(5000):
            if nouns[i] in dic[str(j)]:
                a[j][i] = 1
    return a

matrix=matrix()
#print(matrix)

data1 = pd.DataFrame(matrix)
data1.to_csv('/Users/hhy/Desktop/1/comment.csv')