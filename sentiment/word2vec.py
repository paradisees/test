import csv
import gensim

#训练
def fea(text):
    with open(text,encoding='gb18030') as f:
        rows = csv.reader(f)
        res=[]
        #temp用来存放所有的词
        temp=[]
        for row in rows:
            res1=[]
            for i in range(len(row)):
                if row[i]!='' and row[i]!='.':
                    res1.append(row[i])
                    temp.append(row[i])
            res.append(res1)
        #fea = list(set(res))
        #print(len(temp),list(set(temp)))
        return res,list(set(temp))
#使用所有的paticiple之后的词来进行
m,temp=fea('/Users/hhy/Desktop/raw.csv')

#print(m)
print(len(temp))

'''
for i in range(len(m)):
    if 'aginas' in m[i]:
        print(1)
'''

#model = gensim.models.Word2Vec(m, size=100, min_count=1,workers=4)
#model.save('/Users/hhy/Desktop/model')
model=gensim.models.Word2Vec.load('/Users/hhy/Desktop/model')


'''
words=model.most_similar(positive='writes')
for word,similarity in words:
    print(word,similarity)
'''

with open('/Users/hhy/Desktop/vector1.csv', 'w', encoding='gb18030', newline='') as f:
    writer = csv.writer(f)
    #writer.writerow((model[temp[0]]))
    for i in range(len(temp)):
        try:
            writer.writerow(model[temp[i]])
        except:
            print('不存在哦')
with open('/Users/hhy/Desktop/vector2.csv', 'w', encoding='gb18030', newline='') as f:
    writer = csv.writer(f)
    #writer.writerow((model[temp[0]]))
    for i in range(len(temp)):
        try:
            writer.writerow([temp[i]])
        except:
            print('不存在哦')


