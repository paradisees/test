import numpy as np
from keras.models import Sequential
from keras.layers import Dense,Activation
from keras.optimizers import RMSprop
from sklearn.preprocessing import OneHotEncoder
from sklearn import cross_validation
import csv
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.grid_search import GridSearchCV
from keras.layers import Dropout
from keras import regularizers
from keras.constraints import maxnorm
np.random.seed(1337)  # for reproducibility
BATCH_SIZE = 4
BATCH_INDEX = 0
data=[]
mark=[]
with open('/Users/hhy/Desktop/test/data.csv','r',encoding='utf-8_sig') as f:
    csv_reader=csv.reader(f)
    for x in csv_reader:
        data.append(list(map(float,x[0:-1])))
        mark.append([float(x[-1])])
data=np.array(data).astype(np.float32)
mark=np.array(mark)
'''res=[]
for i in range(10):
    x_train, x_test,y_train,  y_test = cross_validation.train_test_split(
        data, mark, test_size=0.1,random_state=0)
    model = Sequential()
    model.add(Dense(11, input_dim=737, init='glorot_normal', activation='softsign'))
    model.add(Dropout(0.3))
    model.add(Dense(1, init='glorot_normal', activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=3, batch_size=2,verbose=0)
    loss_and_metrics = model.evaluate(x_test, y_test,verbose=0)
    #loss_and_metrics1 = model.evaluate(x_train, y_train,verbose=0)
    res.append(loss_and_metrics[1])
    print ('----------',loss_and_metrics[1])
print('mean',sum(res)/len(res))
#print ('/n','----------',loss_and_metrics1[1])
'''

x_train, x_test,y_train,  y_test = cross_validation.train_test_split(
        data, mark, test_size=0.05,random_state=1)
model = Sequential()
model.add(Dense(11, input_dim=144, init='glorot_normal', activation='softsign'))
model.add(Dropout(0.3))
model.add(Dense(1, init='glorot_normal', activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
max=0
for step in range(10001):
    # data shape = (batch_num, steps, inputs/outputs)
    X_batch = x_train[BATCH_INDEX: BATCH_INDEX+BATCH_SIZE, :]
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