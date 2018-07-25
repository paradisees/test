import sys

'''
import sys
n = int(sys.stdin.readline().strip())
for i in range(n):
    line = sys.stdin.readline().strip()
    res = list(map(int, line.split()))
    break
print(res)

输出防止后面多个空格
print(" ".join(str(num) for num in output))

a = [1,2,3,4]
print(" ".join(str(i) for i in a))
1 2 3 4

#generator
def f(n):
    for i in range(1,n+1):
        for j in range(1,n+1):
            yield pow(i,j)
for value in f(10):
    print(value)
'''

if __name__ == "__main__":
    # 读取第一行的n
    n = int(sys.stdin.readline().strip())
    ans = 0
    res=[]
    result=[]
    for i in range(n):
        # 读取每一行
        line = sys.stdin.readline().strip()
        # 把每一行的数字分隔后转化成int列表
        values = list(map(int, line.split()))
        res.append(values)
    result.append(max(res))
    res.remove(max(res))

    for item in res:
        if item[1]>res[0][1]:
            result.append(item)

    print(result)

