#提取特征集合
def fea(text):
    import csv
    with open(text) as f:
        rows = csv.reader(f)
        res=[]
        for row in rows:
            for i in range(len(row)):
                if row[i]!='':
                    res.append(row[i])
        fea = list(set(res))
        return fea
m=fea('/Users/hhy/Desktop/2/test.csv')
print(m,len(m))
