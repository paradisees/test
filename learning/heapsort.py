from collections import deque
L = deque([49,38,65,97,76,13,27,49])
L.appendleft(0)
def element_exchange(numbers,low,high):
    temp = numbers[low]
    i = low
    j = 2*i

    while j<=high:
        if j<high and numbers[j]<numbers[j+1]:
            j = j+1
        if temp<numbers[j]:
            numbers[i] = numbers[j]
            i = j
            j = 2*i
        else:
            break
    numbers[i] = temp

def top_heap_sort(numbers):
    length = len(numbers)-1
    first_exchange_element = int(length/2)
    print (first_exchange_element)
    for x in range(first_exchange_element):
        element_exchange(numbers,first_exchange_element-x,length)
    for y in range(length-1):
        temp = numbers[1]
        numbers[1] = numbers[length-y]
        numbers[length-y] = temp
        element_exchange(numbers,1,length-y-1)

if __name__=='__main__':
    top_heap_sort(L)
    for x in range(1,len(L)):
        print (L[x],)