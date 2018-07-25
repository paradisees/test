import csv
def row_csv2dict(csv_file):
    dict_club_pos={}
    dict_club_neg={}
    dict_club_neu={}
    with open(csv_file)as f:
        reader=csv.reader(f,delimiter=',')
        for row in reader:
            if row[0]>row[1]:
                dict_club_pos[row[2]]=row[0]
            elif row[0]<row[1]:
                dict_club_neg[row[2]] = row[1]
            else:
                dict_club_neu[row[2]]=row[0]
    return dict_club_pos
dict=row_csv2dict('/Users/hhy/Desktop/test.csv')
print(dict)
res=[]
if 'able' in dict:
    print('1')