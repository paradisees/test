'''
import difflib
def difflib_leven(str1, str2):
    leven_cost = 0
    s = difflib.SequenceMatcher(None, str1, str2)
    for tag, i1, i2, j1, j2 in s.get_opcodes():
        #print('{:7} a[{}: {}] --> b[{}: {}] {} --> {}'.format(tag, i1, i2, j1, j2, str1[i1: i2], str2[j1: j2]))

        if tag == 'replace':
            leven_cost += max(i2-i1, j2-j1)
        elif tag == 'insert':
            leven_cost += (j2-j1)
        elif tag == 'delete':
            leven_cost += (i2-i1)
    return leven_cost
print(difflib_leven('gooooood','gad'))'''
'''
#最长公共子序列
import numpy
def find_lcseque(s1, s2):
    # 生成字符串长度加1的0矩阵，m用来保存对应位置匹配的结果
    m = [[0 for x in range(len(s2) + 1)] for y in range(len(s1) + 1)]
    # d用来记录转移方向
    d = [[None for x in range(len(s2) + 1)] for y in range(len(s1) + 1)]

    for p1 in range(len(s1)):
        for p2 in range(len(s2)):
            if s1[p1] == s2[p2]:  # 字符匹配成功，则该位置的值为左上方的值加1
                m[p1 + 1][p2 + 1] = m[p1][p2] + 1
                d[p1 + 1][p2 + 1] = 'ok'
            elif m[p1 + 1][p2] > m[p1][p2 + 1]:  # 左值大于上值，则该位置的值为左值，并标记回溯时的方向
                m[p1 + 1][p2 + 1] = m[p1 + 1][p2]
                d[p1 + 1][p2 + 1] = 'left'
            else:  # 上值大于左值，则该位置的值为上值，并标记方向up
                m[p1 + 1][p2 + 1] = m[p1][p2 + 1]
                d[p1 + 1][p2 + 1] = 'up'
    (p1, p2) = (len(s1), len(s2))
    print(numpy.array(m))
    print(numpy.array(d))
    s = []
    while m[p1][p2]:  # 不为None时
        c = d[p1][p2]
        if c == 'ok':  # 匹配成功，插入该字符，并向左上角找下一个
            s.append(s1[p1 - 1])
            p1 -= 1
            p2 -= 1
        if c == 'left':  # 根据标记，向左找下一个
            p2 -= 1
        if c == 'up':  # 根据标记，向上找下一个
            p1 -= 1
    s.reverse()
    return ''.join(s)


print(find_lcseque('goopoood','gopd')  )
'''
'''
#最长公共子串
def lcs(s1,s2):
    res=[[0 for i in range(len(s2)+1)] for j in range(len(s1)+1)]
    max=0
    p=0
    for i in range(len(s1)):
        for j in range(len(s2)):
            if s1[i]==s2[j]:
                res[i+1][j+1]=res[i][j]+1
                if res[i+1][j+1]>max:
                    max=res[i+1][j+1]
                    p=i+1
    return s1[p-max:p]
print(lcs('abc','abdsa'))
'''

'''
#floyd算法
import numpy as np

Max = 100
v_len = 4
edge = np.mat([[0, 1, Max, 4], [Max, 0, 9, 2], [3, 5, 0, 8], [Max, Max, 6, 0]])
A = edge[:]
path = [[0 for i in range(v_len)] for j in range(v_len)]
#A为点到点最小值
#path为点到点的前驱路径
def Folyd():
    for i in range(v_len):
        for j in range(v_len):
            if (edge[i, j] != Max and edge[i, j] != 0):
                path[i][j] = i+1

    print('init:')
    print(A, '\n', path)
    for a in range(v_len):
        for b in range(v_len):
            for c in range(v_len):
                if (A[b, a] + A[a, c] < A[b, c]):
                    A[b, c] = A[b, a] + A[a, c]
                    path[b][c] = path[a][c]
    print('result:')
    print(A, '\n', path)
    return path

def Path(path,s,t):
    res=[t]
    new=path[s-1]
    temp=new[t-1]
    for i in range(len(new)):
        if temp!=0 and temp!=s:
            res.append(temp)
            temp=new[temp-1]
        else:
            break
    res.append(s)
    res.reverse()
    print(res)


if __name__ == "__main__":
    path=Folyd()
    Path(path,2,1)'''