
import numpy as np
np.random.seed(1337)  # for reproducibility
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation, Convolution2D, MaxPooling2D, Flatten
from keras.optimizers import Adam
import csv
from sklearn import cross_validation
from keras.applications.resnet50 import ResNet50
from keras.models import Model

data=[]
mark=[]

with open('/Users/hhy/Desktop/test/original.csv','r',encoding='utf-8_sig') as f:
    csv_reader=csv.reader(f)
    for x in csv_reader:
        data.append(list(map(float,x[0:-1])))
        mark.append([float(x[-1])])
#mark = OneHotEncoder().fit_transform(mark).todense().astype(int) #one-hot编码
data=np.array(data).astype(np.float32)
mark=np.array(mark)
x_train, x_test,y_train,  y_test = cross_validation.train_test_split(
    data, mark, test_size=0.05,random_state=1)
x_train=x_train.reshape(-1,737,3,1)
x_test=x_test.reshape(-1,737,3,1)

base_model = ResNet50(input_shape=(None,737,3,1),weights = 'imagenet', include_top = False, pooling = 'avg')
predictions = Dense(1, activation='sigmoid')(base_model.output)
model = Model(inputs=base_model.input, outputs=predictions)

# Another way to define your optimizer
adam = Adam(lr=1e-4)

# We add metrics to get more results you want to see
model.compile(optimizer=adam,
              loss='binary_crossentropy',
              metrics=['accuracy'])

print('Training ------------')
# Another way to train the model
model.fit(x_train, y_train, epochs=200, batch_size=64,)

print('\nTesting ------------')
# Evaluate the model with the metrics we defined earlier
loss, accuracy = model.evaluate(x_test, y_test)

print('\ntest loss: ', loss)
print('\ntest accuracy: ', accuracy)