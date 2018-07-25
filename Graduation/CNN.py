import numpy as np
np.random.seed(1113)  # for reproducibility
from keras.models import Sequential
from keras.layers import *
from keras.optimizers import Adam,SGD
import csv
from sklearn import cross_validation
from keras.layers import Dropout
BATCH_SIZE = 4
BATCH_INDEX = 0
data=[]
mark=[]

with open('/Users/hhy/Desktop/1/test.csv','r',encoding='utf-8_sig') as f:
    csv_reader=csv.reader(f)
    for x in csv_reader:
        data.append(list(map(float,x[0:-1])))
        mark.append([float(x[-1])])
#mark = OneHotEncoder().fit_transform(mark).todense().astype(int) #one-hot编码
data=np.array(data).astype(np.float32)
mark=np.array(mark)
x_train, x_test,y_train,  y_test = cross_validation.train_test_split(
    data, mark, test_size=0.05 ,random_state=0)
x_train=x_train.reshape(-1,8,89,1)
x_test=x_test.reshape(-1,8,89,1)

model = Sequential()
# Conv layer 1 output shape (32, 28, 28)
model.add(Convolution2D(
    batch_input_shape=(None,8,89,1),
    filters=32,
    kernel_size=5,
    strides=1,
    padding='same',     # Padding method
))
model.add(Activation('relu'))

# Pooling layer 1 (max pooling) output shape (32, 14, 14)
model.add(MaxPooling2D(
    pool_size=2,
    strides=2,
    padding='same',    # Padding method
))

# Conv layer 2 output shape (64, 14, 14)
model.add(Convolution2D(64, 5, strides=1, padding='same'))
model.add(Activation('relu'))

# Pooling layer 2 (max pooling) output shape (64, 7, 7)
model.add(MaxPooling2D(2, 2, 'same'))

# Fully connected layer 1 input shape (64 * 7 * 7) = (3136), output shape (1024)
model.add(Flatten())
model.add(Dense(1024))
model.add(Activation('relu'))
model.add(Dropout(0.3))
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.3))
# Fully connected layer 2 to shape (10) for 2 classes
model.add(Dense(1))
model.add(Activation('sigmoid'))

# Another way to define your optimizer
adam = Adam(lr=1e-4)
sgd = SGD(lr=0.05, momentum=0.9, decay=1e-6, nesterov=True)
# We add metrics to get more results you want to see
model.compile(optimizer=adam,
              loss='binary_crossentropy',
              metrics=['accuracy'])
#print(len(model.get_weights()),len(model.get_weights()[-1]))
print(model.summary())
'''
# Another way to train the model
new=model.fit(x_train, y_train, epochs=30, batch_size=2,verbose=0)
print(new.history)
print('\nTesting ------------')
# Evaluate the model with the metrics we defined earlier
loss, accuracy = model.evaluate(x_test, y_test)

print('\ntest loss: ', loss)
print('\ntest accuracy: ', accuracy)'''
max=0
for step in range(5001):
    # data shape = (batch_num, steps, inputs/outputs)
    X_batch = x_train[BATCH_INDEX: BATCH_INDEX+BATCH_SIZE, :, :]
    Y_batch = y_train[BATCH_INDEX: BATCH_INDEX+BATCH_SIZE, :]
    cost = model.train_on_batch(X_batch, Y_batch)
    BATCH_INDEX += BATCH_SIZE
    BATCH_INDEX = 0 if BATCH_INDEX >= x_train.shape[0] else BATCH_INDEX

    if step % 100 == 0:
        cost, accuracy = model.evaluate(x_test, y_test, batch_size=y_test.shape[0], verbose=False)
        cost1, accuracy1 = model.evaluate(x_train, y_train, batch_size=y_test.shape[0], verbose=False)
        print('test cost: ', cost, 'test accuracy: ', accuracy)
        if accuracy>max:
            max=accuracy
        print('train cost: ', cost1, 'train accuracy: ', accuracy1)
        #print(len(model.get_weights()),len(model.get_weights()[1]))
print(max)