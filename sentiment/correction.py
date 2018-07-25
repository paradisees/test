from hunspell import Hunspell
import csv
h=Hunspell()
def correction(num):
    if h.spell(str(num)) is False:
        new=h.suggest(str(num))[0]
        return new
    else:
        return num
res=[]
with open('/Users/hhy/Desktop/raw1.csv') as csvfile1:
    rows = csv.reader(csvfile1)
    with open('/Users/hhy/Desktop/write.csv','a', encoding='utf-8' , newline='') as f:
        writer = csv.writer(f)
        for row in rows:
            res = []
            for num in row:
                try:
                    if num == ' ':
                        continue
                    else:
                        res.append(correction(num))
                except:
                    continue
            writer.writerow(res)
