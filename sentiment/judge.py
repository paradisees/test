import csv
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
res1=fea('/Users/hhy/Desktop/2/participle.csv')
res2=fea('/Users/hhy/Desktop/3/test.csv')
print(len(res2))
n=0
result=[]
for res in res2:
    if res not in res1:
        n+=1
        result.append(res)
print(n)
print(result)
with open('/Users/hhy/Desktop/result.csv', 'w', encoding='gb18030', newline='') as f:
    writer = csv.writer(f)
    for res in result:
        writer.writerow([res])