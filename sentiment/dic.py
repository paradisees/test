import csv
def row_csv2dict(csv_file):
    dict={}
    with open(csv_file,encoding='utf-8_sig')as f:
        reader=csv.reader(f,delimiter=',')
        for row in reader:
            item=row[1].split( )
            n = len(item)
            for i in range(n):
                dict[item[i].split('#')[0]] = row[0]
        #for key in dict.keys():
            #dict[key]=float(dict[key][0])-float(dict[key][1])
    return dict
dic=row_csv2dict('/Users/hhy/Desktop/sentiment1.csv')
#print(dic)
print(dic['good'])
#str='speaks'
#print(float(dic[str][0])-float(dic[str][1]))

