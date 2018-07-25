#!/usr/bin/python
# -*- coding: UTF-8 -*-
'''排序
l=[]
for i in range(3):
    x=int (raw_input())
    l.append(x)
l.sort()
print(l)
'''


'''斐波那契
def fib(n):
    i,j=1,1
    for a in range(n-1):
        temp=i
        i=j
        j=j+temp
        print(i,j)
    return i
print fib(10)

def fib(n):
    if n == 1:
        return [1]
    if n == 2:
        return [1, 1]
    fibs = [1, 1]
    for i in range(2, n):
        fibs.append(fibs[-1] + fibs[-2])
    return fibs

# 输出前 10 个斐波那契数列
print fib(10)
'''

'''九九
for i in range(1,10):

    for j in range(1,i+1):
        print "%d * %d=%d" % (i,j,i*j),
    #注意这儿的逗号，下面""表示换行
    print ""
'''


'''延迟输出
import time
hhy={1:"h",2:"hy"}
for key,value in dict.items(hhy):
    print (key,value)
    time.sleep(5)
'''


'''
import time
print time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))
time.sleep(2)
print time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))
'''


'''素数
import math
count=0
#注意leap的引入
leap=1
for i in range(101,201):
    for j in range(2,int(math.sqrt(i)+1)):
        if i%j==0:
            leap=0
            break
#注意下一个if句的位置
    if leap==1:
        print("%d 为素数" %i)
        count+=1
    leap=1
print(count)
'''


'''水仙花数 三个for的循环 注意立方的表示
for n in range(100,1000):
    for i in range(0,10):
        for j in range(0,10):
            for k in range(0, 10):
                if n==i*100+j*10+k and n==i**3+j**3+k**3:
                    print("da",i,j,k)
    改良版
for n in range(100,1000):
    i = n / 100
    j = n / 10 % 10
    k = n % 10
    if n == i ** 3 + j ** 3 + k ** 3:
        print n
        '''

'''数的分解 while循环保证每次都可以从2开始遍历，最后除到n为1的时候n=i
print("请输入要分解的数")
n=int(raw_input())
print"%d=" % n,
if not isinstance(n, int) or n <= 0 :
    print '请输入一个正确的数字 !'
    exit(0)
if (n==1):
    print("1=1*1")
while n != 1:
    for i in range(2,n+1):
        if n%i==0:
            n=n/i
            if(n==1):
                print(i)
            else:print '%d *' % i,
            break
'''

'''判断数字字符啥的数量  用string，
import string
s=raw_input("输入字符串")
#print("string.count(c):",string.count(c))
alpha=0
space=0
digit=0
others=0
for c in s:
    if c.isalpha():
        alpha+=1
    elif c.isdigit():
        digit+=1
    elif c.isspace():
        space+=1
    else:others+=1
print("alpha = %d,space = %d,digit = %d,others = %d" % (alpha,space,digit,others))
'''


'''输出如下，注意num和total分别使用  也可以考虑Sn=[]，例如Sn最后为[2,22],然后reduce和dba Sn
4+44+444+4444+44444
i=int(raw_input("输入想计算的数"))
j=int(raw_input("输入计算的轮数"))
total=0
num=0
for n in range(1,j+1):
    num=i*(10**(n-1))+num
    total+=num
print(total)
'''
'''方法二
i=int(raw_input("输入想计算的数"))
j=int(raw_input("输入计算的轮数"))
Sn=[]
num=0
for n in range(1,j+1):
    num=i*(10**(n-1))+num
    Sn.append(num)
print reduce(lambda x,y:x+y, Sn)
'''
#reduce函数和lambda表达式组合应用：(下式中为加法运算) labdba为精简的函数表达方法
'''list=[1,2,3,4,5]
print reduce(lambda x,y:x+y,list)

def ds(x):
    return 2*x+1
print ds(2)

g=lambda x:x*2+1
print g(5)
'''
''' ？？？？？
def qu(n):
    ln=[]
    while n!=1:
        for i in range(2,n+1):
            if n%i==0 :
              #  n=n/i
                ln.append(i)
   # print ln
    total=reduce(lambda x,y:x+y,ln)
    if 1+total==i:
        print "%d是全数" % i

for i in range(2,1001):
    qu(i)
'''

'''猴子吃桃
total=1
sum1=[]
for i in range(1,11):
    sum1.append(total)
    total=int((total+1)*2)
print sum1[9]
'''
'''
x2 = 1
for day in range(9,0,-1):
    x1 = (x2 + 1) * 2
    x2 = x1
print x1
'''

''' 打印菱形 stdout.write默认不换行，并且第二个for写的东西会跟着第一个写完的信息后面写最后用个print换行
from sys import stdout
for i in range(4):
    for j in range(2 - i + 1):
        stdout.write(' ')
    for k in range(2 * i + 1):
        stdout.write('*')
    print

for i in range(3):
    for j in range(i + 1):
        stdout.write(' ')
    for k in range(4 - 2 * i + 1):
        stdout.write('*')
    print
    '''
'''注意用a,b=a+b,a替换temp=a,a=a+b,b=temp
ln=[]
x=0.0
a=2.0
b=1.0
for i in range(1,21):
    x=a/b
    a,b=a+b,a
    print(x)
    ln.append(x)
print reduce(lambda x,y:x+y,ln)
'''

'''递归方法求阶乘
def fact(j):
    if j==0:
        sum=1
    else:
        sum=j*fact(j-1)
    return sum
for i in range(5):
    print '%d! = %d' % (i,fact(i))
    '''


'''逆序输出
def output(s,l):
    if l==0:
        return
    print s[l-1],
    output(s,l-1)

s=raw_input("输入")
l=len(s)
print l
output(s,l)
'''

'''回数判断  注意int和str类型的转换
i=int(raw_input("输入五位数"))
if not isinstance(i,int) or i<9999 or i>100000:
    print "输入有误"
    exit(0)
str1=str(i)
if str1[0]==str1[4] and str1[1]==str1[3]:
    print "yes"
else:print "no"
'''



'''倒序输出
a = ['one', 'two', 'three']
for i in a[::-1]:
	print i
'''

'''按符号分割列表
L = ['1','2','3','4','5']
print ','.join(L)

S=('1','3','2','4')
print ':'.join(S)

L = [1,2,3,4,5]
s1 = ','.join(str(n) for n in L)
print s1
'''

'''建立二维数组
a = []
sum = 0.0
for i in range(3):
    a.append([])
    for j in range(3):
        a[i].append(float(raw_input("input num:\n")))
for i in range(3):
    sum += a[i][i]
print sum
print a[1][1]
'''


'''
#冒泡排序：两两比较，大的往下沉，第一轮保证最大的数在最下面
#添加一个数进去按原规则排序
s=[]
for m in range(6):
    i=int(raw_input("输入数"))
    s.append(i)
n=int(raw_input("要插入的数为"))
s.append(n)
print s
for j in range(0,len(s)):
    for k in range(0,len(s)-1):
        if s[k]>s[k+1]:
            temp=s[k]
            s[k]=s[k+1]
            s[k+1]=temp
print s

''''''函数法
def pai(s):
    for j in range(0, len(s)):
        for k in range(0, len(s) - 1):
            if s[k] > s[k + 1]:
                temp = s[k]
                s[k] = s[k + 1]
                s[k + 1] = temp
    print s

s1=[]
for m in range(6):
    i=int(raw_input("输入数"))
    s1.append(i)
pai(s1)

n=int(raw_input("要插入的数为"))
s1.append(n)
pai(s1)
'''

'''作为类的属性
class Static:
    StaticVar = 5
    def varfunc(self):
        self.StaticVar += 1
        print self.StaticVar

print Static.StaticVar
a = Static()
for i in range(3):
    a.varfunc()
'''

'''
#生成 10 到 20 之间的随机数
import random
print random.uniform(10, 20)
'''

'''画圆
if __name__ == '__main__':
    from Tkinter import *

    canvas = Canvas(width=800, height=600, bg='yellow')
    canvas.pack(expand=YES, fill=BOTH)
    k = 1
    j = 1
    for i in range(0,26):
        canvas.create_oval(310 - k,250 - k,310 + k,250 + k, width=1)
        k += j
        j += 0.3

    mainloop()
    '''


'''杨辉三角形，引入stdout模块，初始化二维数组全为0
a=[]
for i in range(10):
    a.append([])
    for j in range(10):
        a[i].append(0)
for i in range(10):
    a[i][0]=1
    a[i][i]=1
for i in range(2,10):
    for j in range(1,i):
        a[i][j]=a[i-1][j]+a[i-1][j-1]
from sys import stdout
for i in range(10):
    for j in range(i+1):
        stdout.write(str(a[i][j]))
        stdout.write(' ')
    print
'''

'''有n个整数，使其前面各数顺序向后移m个位置，最后m个数变成最前面的m个数
    同时移动三个数可能输导致赋值错误，所以一个一个移动
def move(array,n,m):
    array_end=array[n-1]
    for i in range(n-1,-1,- 1):
        array[i]=array[i-1]
    array[0]=array_end
    m-=1
    if m>0 :move(array,n,m)
'''


#有n个人围成一圈，顺序排号。从第一个人开始报数（从1到3报数），凡报到3的人退出圈子，问最后留下的是原来第几号的那位。
#用一个标志位k，每数一个数，标志位加1，当标志位为3时，那个人的数置为0，重置为0；另一个标志位i，用于
#判断一轮是否走完，当走完的时候，i重置为0，重新开始这圈；最后输出整个数列，输出不为0的那个
#另一个标志位m用于保证循环，每淘汰一个数，m加1，当m为n-1时，跳出循环
'''
if __name__ == '__main__':
    nmax = 50
    n = int(raw_input('请输入总人数:'))
    num = []
    for i in range(n):
        num.append(i + 1)

    i = 0
    k = 0
    m = 0

    while m < n - 1:
        if num[i] != 0 :
            k += 1
        if k == 3:
            num[i] = 0
            k = 0
            m += 1
        i += 1
        if i == n :
            i = 0

    i = 0
    while num[i] == 0:
        i += 1
    print num[i]
'''


#二维数组
'''
student = []
for i in range(2):
    student.append(['',''])
for j in range(2):
    student[j][0] = raw_input('input student num1:\n')
    student[j][1] = raw_input('input student num2:\n')
print student[1][1]


a = []
for i in range(3):
    a.append([])
    for j in range(3):
        a[i].append(float(raw_input("input num:\n")))
print a
'''


'''
a=5
print a*'*'
'''


'''文件写入直至输入#
if __name__ == '__main__':
    from sys import stdout
    filename = raw_input('input a file name:\n')
    fp = open(filename,"w")
    ch = raw_input('input string:\n')
    while ch != '#':
        fp.write(ch)
        stdout.write(ch)
        ch = raw_input('')
    fp.close()
    '''

'''从两个文件中读取内容后合并到文件c中
if __name__ == '__main__':
    import string
    fp = open('test1.txt')
    a = fp.read()
    fp.close()

    fp = open('test2.txt')
    b = fp.read()
    fp.close()

    fp = open('test3.txt','w')
    l = list(a + b)
    l.sort()
    s = ''
    s = s.join(l)
    fp.write(s)
    fp.close()
    '''


#快速排序
'''
def quickSort(L, low, high):
    i = low
    j = high
    if i >= j:
        return L
    key = L[i]
    while i < j:
        while i < j and L[j] >= key:
            j = j-1
        L[i],L[j]=L[j],L[i]
        while i < j and L[i] <= key:
            i = i+1
        L[i], L[j] = L[j], L[i]
    quickSort(L, low, i-1)
    quickSort(L, j+1, high)
    return L

l=[6,2,7,3,8,5,9]

quickSort(l,0,6)
print l
'''



#合并排序
'''
def loop_merge_sort(l1,l2):
    tmp=[]
    while len(l1) > 0 and len(l2) > 0:
        if l1[0] < l2[0]:
            tmp.append(l1[0])
            del l1[0]
        else:
            tmp.append(l2[0])
            del l2[0]
    tmp.extend(l1)
    tmp.extend(l2)
    return tmp

l1=[1,3,4,5]
l2=[2,3,6,8]
print (loop_merge_sort(l1, l2))
'''