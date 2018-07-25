#分词
import csv
def textParse(bigString):
    import re
    from nltk.corpus import stopwords
    listOfTokens = re.split(r'\W*', bigString)
    text= [tok.lower() for tok in listOfTokens if len(tok) > 2]
    filtered = ''
    res=[]
    for w in text:
        if w not in stopwords.words('english'):
            res.append(filtered.join(w))
    return res

with open('/Users/hhy/Desktop/3/test1.csv') as csvfile1:
    rows = csv.reader(csvfile1)
    with open('/Users/hhy/Desktop/3/test.csv','w', encoding='gb18030' , newline='') as f:
        writer = csv.writer(f)
        res1=[]
        for row in rows:
            if textParse(str(row[1]))!=[]:
                writer.writerow(textParse(str(row[1])))
            else:
                res1.append(row[0])
print(res1,len(res1))