'''
class Solution(object):
    def hammingDistance(self, x, y):
        return bin(x^y).count('1')
'''

'''
class Solution(object):
    def nextGreaterElement(self, findNums, nums):
        d = {}
        st = []
        ans = []

        for x in nums:
            while len(st) and st[-1] < x:
                #给字典中的值赋值
                d[st.pop()] = x
            st.append(x)
        print st
        for x in findNums:
            #若中在字典存在x，返回对应值；不存在返回-1
            ans.append(d.get(x, -1))
        return ans
findNums=[4,1,2]
nums=[1,3,4,2]
g=Solution()
print g.nextGreaterElement(findNums,nums)
'''

#小岛递归
'''
class Solution(object):
    def islandPerimeter(self, grid):
        if len(grid)==0:
            return 0
        for i in range(0,len(grid)):
            for j in range(0,len(grid[i])):
                if grid[i][j]==1:
                    return self.compute(grid,i,j)
        return 0

    def compute(self,grid,i,j):
        count=0
        grid[i][j]=-1
        if i-1<0 or grid[i-1][j]==0:
            count+=1
        elif grid[i-1][j]==1:
            count+=self.compute(grid,i-1,j)
        if j-1<0 or grid[i][j-1]==0:
            count+=1
        elif grid[i][j-1]==1:
            count+=self.compute(grid,i,j-1)
        if j+1>=len(grid[i]) or grid[i][j+1]==0:
            count+=1
        elif grid[i][j+1]==1:
            count+=self.compute(grid,i,j+1)
        if i+1>=len(grid) or grid[i+1][j]==0:
            count+=1
        elif grid[i+1][j]==1:
            count+=self.compute(grid,i+1,j)
        return count

grid1 = []
for i in range(4):
    grid1.append([])
    for j in range(4):
        grid1[i].append(int(raw_input("input num:\n")))
print grid1
g=Solution()
g.islandPerimeter(grid1)
'''