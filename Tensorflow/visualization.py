import numpy as np
import struct
import matplotlib.pyplot as plt

filename = '/Users/hhy/datasets/MNIST_data/train-images-idx3-ubyte'
binfile = open(filename , 'rb')
buf = binfile.read()

index = 0
magic, numImages , numRows , numColumns = struct.unpack_from('>IIII' , buf , index)
index += struct.calcsize('>IIII')

im = struct.unpack_from('>1568B' ,buf, index)
index += struct.calcsize('>1568B')

im = np.array(im)
im = im.reshape(56,28)

fig = plt.figure()
plotwindow = fig.add_subplot(111)
plt.imshow(im , cmap='gray')
plt.show()