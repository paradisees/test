#冒泡，从后往前，小的浮上去
def bubble_sort(lists):
    count = len(lists)
    for i in range(0, count):
        for j in range(i + 1, count):
            if lists[i] > lists[j]:
                lists[i], lists[j] = lists[j], lists[i]
    return lists


#快速排序 nlogn
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
print (l)


#归并排序nlogn
def MergeSort(lists):
    if len(lists) <= 1:
        return lists
    num = int(len(lists)/2 )
    left = MergeSort(lists[:num])
    right = MergeSort(lists[num:])
    return Merge(left, right)
def Merge(left,right):
    r, l=0, 0
    result=[]
    while l<len(left) and r<len(right):
        if left[l] < right[r]:
            result.append(left[l])
            l += 1
        else:
            result.append(right[r])
            r += 1
    result += right[r:]
    result+= left[l:]
    return result
print (MergeSort([1, 2,90, 21, 3, 5, 6,4, 7, 23, 45]))


#直接选择排序
def select_sort(lists):
    count = len(lists)
    for i in range(0, count):
        min = i
        for j in range(i + 1, count):
            if lists[min] > lists[j]:
                min = j
        lists[min], lists[i] = lists[i], lists[min]
    return lists