import numpy as np
np.random.seed(1337)  # for reproducibility
from sklearn import cross_validation
from keras.models import Sequential
from keras.layers import *
from keras.optimizers import Adam
from keras.layers import Dropout
import csv
from keras.constraints import maxnorm
from keras import regularizers
from keras.callbacks import EarlyStopping
#712
#144
TIME_STEPS = 8    # same as the height of the image
INPUT_SIZE = 18     # same as the width of the image
BATCH_SIZE = 4
BATCH_INDEX = 0
OUTPUT_SIZE = 1
CELL_SIZE = 40
LR = 0.0001
data=[]
mark=[]
with open('/Users/hhy/Desktop/test/data.csv','r',encoding='utf-8_sig') as f:
    csv_reader=csv.reader(f)
    for x in csv_reader:
        data.append(list(map(float,x[0:-1])))
        mark.append([float(x[-1])])
data=np.array(data).astype(np.float32)
mark=np.array(mark)
x_train, x_test,y_train,  y_test = cross_validation.train_test_split(
    data, mark, test_size=0.05,random_state=1)
x_train=x_train.reshape(-1,8,18)
x_test=x_test.reshape(-1,8,18)

# build RNN model
model = Sequential()

# RNN cell
'''model.add(SimpleRNN(
    batch_input_shape=(None, TIME_STEPS, INPUT_SIZE),       # Or: input_dim=INPUT_SIZE, input_length=TIME_STEPS,
    output_dim=CELL_SIZE,
    unroll=True,
    kernel_regularizer=regularizers.l2(0.01)
))'''
model.add(LSTM(
    batch_input_shape=(None, TIME_STEPS, INPUT_SIZE),
    output_dim=CELL_SIZE,
    unroll=True,
    kernel_regularizer=regularizers.l2(0.01)))
model.add(Dropout(0.5))

# output layer
model.add(Dense(OUTPUT_SIZE))
model.add(Activation('sigmoid'))
# optimizer
adam = Adam(LR)
model.compile(optimizer=adam,
              loss='binary_crossentropy',
              metrics=['accuracy'])
'''
early_stopping = EarlyStopping(monitor='val_loss', patience=2)
hist = model.fit(x_train, y_train, validation_data=(x_test, y_test), verbose=0,callbacks=[early_stopping],epochs=30, batch_size=2,)
print(hist.history)
loss_and_metrics = model.evaluate(x_test, y_test, verbose=0)
print('----------', loss_and_metrics[1])'''
max=0
# training
for step in range(10001):
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
print(max)