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
print(m)
