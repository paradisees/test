class TreeNode:
    def __init__(self, x):
         self.val = x
         self.left = None
         self.right = None
class Solution:
    def IsBalanced_Solution(self, pRoot):
        # write code here
        if not pRoot:
            return 1

        left_depth = self.IsBalanced_Solution(pRoot.left)
        right_depth = self.IsBalanced_Solution(pRoot.right)
        if not left_depth or not right_depth:
            return False

        if abs(left_depth - right_depth) > 1:
            return False

        return max([left_depth, right_depth]) + 1
node1=TreeNode(1)
node2=TreeNode(2)
node3=TreeNode(3)
node4=TreeNode(4)
node5=TreeNode(5)
node6=TreeNode(6)
node7=TreeNode(7)
node8=TreeNode(8)
node9=TreeNode(9)
node1.left=node2
node1.right=node3
node2.left=node4
node2.right=node5
node5.left=node6
node6.left=node7
node3.right=node8
node8.right=node9
root=Solution()
print(root.IsBalanced_Solution(node1))