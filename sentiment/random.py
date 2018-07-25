import random
res=[]
text=open('/Users/hhy/Desktop/3/-11.csv',encoding='utf-8_sig')
target=open('/Users/hhy/Desktop/3/-111.csv','w')
lines=text.readlines()
m=0
n=len(lines)
while len(res)<12390:
    s= random.randint(0, n - 1)
    res.append(lines[s])
for content in res:
    target.write(content)