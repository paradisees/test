import importlib
import sys

importlib.reload(sys)
import urllib.request
import json  # 导入json模块
import hashlib
import urllib
from sentiment import random
import csv

def translate(word):
    appid = '20170822000075837'
    secretKey = '9aoSvl5YVygaGvuTUp9_'
    myurl = 'http://api.fanyi.baidu.com/api/trans/vip/translate'
    fromLang = 'auto'
    toLang = 'en'
    salt = random.randint(32768, 65536)
    sign = appid + word + str(salt) + secretKey
    sign = hashlib.md5(sign.encode()).hexdigest()
    myurl = myurl + '?appid=' + appid + '&q=' + urllib.parse.quote(
        word) + '&from=' + fromLang + '&to=' + toLang + '&salt=' + str(salt) + '&sign=' + sign
    resultPage = urllib.request.urlopen(myurl)
    resultJason = resultPage.read().decode('utf-8')
    try:
        js = json.loads(resultJason)  # 将json格式的结果转换成Python的字典结构
        dst = str(js["trans_result"][0]["dst"])  # 取得翻译后的文本结果
        if dst[0]:
            outDst = dst.strip() + "\n"
            res.append(outDst)
    except Exception as e:   # 如果翻译出错，则输出原来的文本
        print(e)
res=[]
with open('/Users/hhy/Desktop/try.csv','r') as csvfile:
    reader = csv.reader(csvfile)
    column = [row[0] for row in reader]

if __name__ == '__main__':
    for i in column:
        translate(i)
#print(res)

with open('/Users/hhy/Desktop/try.csv') as csvfile1:
    rows = csv.reader(csvfile1)
    with open('/Users/hhy/Desktop/try1.csv','w', encoding='gb18030',newline='') as f:
        writer = csv.writer(f)
        i=0
        for row in rows:
            row.append([column[i]],[res[i]])
            #print(row)
            writer.writerow(row)
            i+=1