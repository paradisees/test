import csv
import pandas as pd
def row_csv2dict(csv_file):
    dict={}
    with open(csv_file,encoding='utf-8_sig')as f:
        reader=csv.reader(f,delimiter=',')
        for row in reader:
            item=row[1].split( )
            n = len(item)
            for i in range(n):
                dict[item[i].split('#')[0]] = row[0]
    return dict
dic=row_csv2dict('/Users/hhy/Desktop/sentiment1.csv')
#print(dic)

score=[]
with open('/Users/hhy/Desktop/1/eng1.csv') as csvfile1:
    rows = csv.reader(csvfile1)
    for row in rows:
        sum=0.0
        for i in range(len(row)):
            if row[i] in dic:
                sum+=float(dic[row[i]])
        score.append(sum)
#print(score)
data1 = pd.DataFrame(score)
data1.to_csv('/Users/hhy/Desktop/1/score1.csv')



