#-*- coding:UTF-8 -*-
'''全排列
str=['a','b','c','\n']
def P(str,i):
    if str==None:
        return
    if str[i]=='\n':
        print "%s\n" %str
    else:
        for j in range(i,len(str)-1):
            str[j],str[i]=str[i],str[j]
            P(str,i+1)
            str[j], str[i] = str[i], str[j]
P(str,0)
'''

#广度优先和深度优先搜索
'''
class Graph(object):

    def __init__(self,*args,**kwargs):
        self.node_neighbors = {}
        self.visited = {}
    def add_nodes(self,nodelist):
        for node in nodelist:
            self.add_node(node)
    def add_node(self,node):
        if not node in self.nodes():
            self.node_neighbors[node] = []
    def add_edge(self,edge):
        u,v = edge
        if(v not in self.node_neighbors[u]) and ( u not in self.node_neighbors[v]):
            self.node_neighbors[u].append(v)

            if(u!=v):
                self.node_neighbors[v].append(u)
    def nodes(self):
        return self.node_neighbors.keys()

    def depth_first_search(self,root=None):
        order = []
        def dfs(node):
            self.visited[node] = True
            order.append(node)
            for n in self.node_neighbors[node]:
                if not n in self.visited:
                    dfs(n)
        if root:
            dfs(root)
        for node in self.nodes():
            if not node in self.visited:
                dfs(node)
        print order
        self.visited = {}
        return order

    def breadth_first_search(self,root=None):
        queue = []
        order = []
        def bfs():
            while len(queue)> 0:
                node  = queue.pop(0)
                self.visited[node] = True
                #print self.visited
                for n in self.node_neighbors[node]:
                    if (not n in self.visited) and (not n in queue):
                        queue.append(n)
                        order.append(n)
        if root:
            queue.append(root)
            order.append(root)
            bfs()
        for node in self.nodes():
            if not node in self.visited:
                queue.append(node)
                order.append(node)
                bfs()
        print order
        self.visited={}
        return order

if __name__ == '__main__':
    g = Graph()
g.add_nodes([i+1 for i in range(8)])
g.add_edge((1, 2))
g.add_edge((1, 3))
g.add_edge((2, 4))
g.add_edge((2, 5))
g.add_edge((4, 8))
g.add_edge((5, 8))
g.add_edge((3, 6))
g.add_edge((3, 7))
g.add_edge((6, 7))
print "nodes:", g.nodes()

order = g.breadth_first_search(1)
order = g.depth_first_search(1)
'''


#最大字段和动态规划算法
'''
def maxSubArray(nums):
    """
    :type nums: List[int]
    :rtype: int
    """
    res = -2 ** 32
    temp = 0
    for num in nums:
        if (temp < 0):
            temp = 0
        temp += num
        if (temp > res):
            res = temp

    return max(temp, res)
print(maxSubArray([-1,-2,-3]))
'''

'''
#-*- coding:utf-8-*-
#0-1背包
def bag(n, c, w, v):
    res = [[-1 for j in range(c + 1)] for i in range(n + 1)]
    print res
    for j in range(c + 1):
        res[0][j] = 0
    for i in range(1, n + 1):
        for j in range(1, c + 1):
            res[i][j] = res[i - 1][j]
            if j >= w[i - 1] and res[i][j] < res[i - 1][j - w[i - 1]] + v[i - 1]:
                res[i][j] = res[i - 1][j - w[i - 1]] + v[i - 1]
    return res


def show(n, c, w, res):
    print('最大价值为:%d' %res[n][c])
    x = [False for i in range(n)]
    j = c
    for i in range(1, n + 1):
        if res[i][j] > res[i - 1][j]:
            x[i - 1] = True
            j -= w[i - 1]
    print('选择的物品为:')
    for i in range(n):
        if x[i]:
            print('第%d个' % i)
            print('')

if __name__ == '__main__':
    n = 5
    c = 10
    w = [2, 2, 6, 5, 4]
    v = [6, 3, 5, 4, 6]
    res = bag(n, c, w, v)
    show(n, c, w, res)
    '''