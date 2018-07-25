# -*- coding:utf-8 -*-
'''
import urllib
import urllib2
import re
import sys

reload(sys)
sys.setdefaultencoding('utf8')
# 处理页面标签类
class Tool:
    # 去除img标签,7位长空格
    removeImg = re.compile('<img.*?>| {7}|')
    # 删除超链接标签
    removeAddr = re.compile('<a.*?>|</a>')
    # 把换行的标签换为\n
    replaceLine = re.compile('<tr>|<div>|</div>|</p>')
    # 将表格制表<td>替换为\t
    replaceTD = re.compile('<td>')
    # 把段落开头换为\n加空两格
    replacePara = re.compile('<p.*?>')
    # 将换行符或双换行符替换为\n
    replaceBR = re.compile('<br><br>|<br>')
    # 将其余标签剔除
    removeExtraTag = re.compile('<.*?>')
    #剔除&nbsp
    removeNbsp=re.compile('&nbsp;|&ldquo;|rdquo;')



    def replace(self, x):
        x = re.sub(self.removeImg, "", x)
        x = re.sub(self.removeAddr, "", x)
        x = re.sub(self.replaceLine, "\n", x)
        x = re.sub(self.replaceTD, "\t", x)
        x = re.sub(self.replacePara, "\n    ", x)
        x = re.sub(self.replaceBR, "\n", x)
        x = re.sub(self.removeExtraTag, "", x)
        x = re.sub(self.removeNbsp," ",x)
        # strip()将前后多余内容删除
        return x.strip()

class CYYZ:
    def __init__(self):
        self.pageIndex=1
        self.tool=Tool()
        self.defaultTitle = "长阳一中"
        self.baseUrl='http://www.cyyz.com.cn'

    #传入页码，获取该页的代码
    def getPage(self,pageIndex):
        try:
            url='http://www.cyyz.com.cn/col/col3002'
            request=urllib2.Request(url)
            response=urllib2.urlopen(request)
            pageCode=response.read().decode('utf-8')
            return pageCode

        except urllib2.URLError, e:
            if hasattr(e, "reason"):
                print "连接失败,错误原因", e.reason
                return None

    #传入整页代码，获得新闻链接
    def getContent(self,pageIndex):
        pageCode=self.getPage(pageIndex)
        if not pageCode:
            print "页面加载失败...."
            return None
        #用\标记转义字符'
        pattern=re.compile('<td><a style.*?href=\'(.*?)\'.*?</td>',re.S)
        items=re.findall(pattern,pageCode)
        list=[]
        number=0
        for item in items:
            list1=self.tool.replace(item)+'\n'
            list.append(list1.decode('utf-8'))
            #print list
            number+=1
        return list,number
        #print list[0]
        #print number

    #获得每条新闻的链接后，读取每条新闻
    def getNews(self,listNum):
        try:
            url=self.baseUrl + str(listNum)
            #print url
            request=urllib2.Request(url)
            response=urllib2.urlopen(request)
            pageCodeNews=response.read().decode('utf-8')
            return pageCodeNews

        except urllib2.URLError, e:
            if hasattr(e, "reason"):
                print "连接失败2,错误原因", e.reason
                return None

    def getNewsContent(self,listNum):
        pageCodeNews=self.getNews(listNum)
        if not pageCodeNews:
            print "页面加载失败...."
            return None
        pattern = re.compile('<title>(.*?)</title>.*?<meta name="author" content="(.*?)">.*?<meta name="pubDate" content="(.*?)">.*?<p><span style="font-size.*?>(.*?)</span></p>',re.S)
        items=re.findall(pattern,pageCodeNews)
        contents=[]
        for item in items:
            #content=self.tool.replace(item)
            #print item[3]
            contents.append([item[0],item[1],item[2],item[3]])
        return contents

    def getOneNews(self,listNum):
        contents=self.getNewsContent(listNum)
        for content in contents:
            content[3]= self.tool.replace(content[3])
            print "标题：%s\n发布人:%s\n时间:%s\n内容:%s" % (content[0], content[1], content[2], content[3])

    # 将新闻写入到文件中
    def writeData(self, contents):
        for i in range(4):
            floorLine = "\n" +"-----------------------------------------------------------------------------------------\n"
            self.file.write(floorLine)
            self.file.write(contents[i])



    def start(self):
        pageCode=self.getPage(1)
        number=self.getContent(1)[1]
        self.file = open(self.defaultTitle + ".txt", "w+")
        #list=self.getContent(1)[0][1]
        #print list
        #print number
        try:
            for i in range(1,number+1):
                print "正在写入第" + str(i) + "个新闻"
                list=self.getContent(1)[0][i-1]
                #print list
                pageCodeNews=self.getNews(list)
                #contents=self.getNewsContent(list)
                #self.getOneNews(list)
                contents1=self.getOneNews(list)
                self.writeData(contents1)
        except IOError, e:
            print "写入异常，原因" + e.message
        finally:
            print "写入任务完成"
cyyz=CYYZ()
cyyz.start()

'''