#提取rawdata中的各种特征信息并分词
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
res=[]
with open('/Users/hhy/Desktop/1/rawdata.csv','r') as csvfile:
    reader = csv.reader(csvfile)
    column = [row[19] for row in reader]

if __name__ == '__main__':
    for i in column:
        res.append(textParse(i))
print(res)


with open('/Users/hhy/Desktop/1/rawdata.csv') as csvfile1:
    rows = csv.reader(csvfile1)
    with open('/Users/hhy/Desktop/1/rawdata1.csv','w', encoding='gb18030',newline='') as f:
        writer = csv.writer(f)
        i=0
        for row in rows:
            if textParse(str(row[19])) != []:
                row.append(res[i])
            #print(row)
                writer.writerow(row)
                i+=1
