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
from keras.constraints import maxnorm

data=[]
mark=[]
with open('/Users/hhy/Desktop/1/test.csv','r',encoding='utf-8_sig') as f:
    csv_reader=csv.reader(f)
    for x in csv_reader:
        data.append(list(map(float,x[0:-1])))
        mark.append([float(x[-1])])
#mark = OneHotEncoder().fit_transform(mark).todense().astype(int) #one-hot编码
x_train, x_test,y_train,  y_test = cross_validation.train_test_split(
    data, mark, test_size=0.05,random_state=1)
def create_model(neurons=1):
    model = Sequential()
    model.add(Dense(neurons, input_dim=737, init='glorot_normal', activation='softsign',W_constraint=maxnorm(1)))
    model.add(Dropout(0.3))
    model.add(Dense(1, init='glorot_normal', activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
model = KerasClassifier(build_fn=create_model, verbose=0,nb_epoch=3, batch_size=2)
'''
batch_size = [j for j in range(2,30,2)]
epochs = [i for i in range(1,10,1)]
param_grid = dict(batch_size=batch_size, nb_epoch=epochs)'''
neurons = [i for i in range(1,151,10)]
param_grid = dict(neurons=neurons)
'''
optimizer = ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']
param_grid = dict(optimizer=optimizer)'''
'''
init_mode = ['uniform', 'lecun_uniform', 'normal', 'zero', 'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform']
param_grid = dict(init_mode=init_mode)'''
'''activation = ['softmax', 'softplus', 'softsign', 'relu', 'tanh', 'sigmoid', 'hard_sigmoid', 'linear']
param_grid = dict(activation=activation)'''
'''weight_constraint = [1, 2, 3, 4, 5]
dropout_rate = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
param_grid = dict(dropout_rate=dropout_rate, weight_constraint=weight_constraint)'''
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1,scoring='accuracy',cv=10)
grid_result = grid.fit(data, mark)
# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
for params, mean_score, scores in grid_result.grid_scores_:
    print("%f (%f) with: %r" % (scores.mean(), scores.std(), params))

'''
model.fit(x_train, y_train, epochs=50, batch_size=10,verbose=0)
loss_and_metrics = model.evaluate(x_test, y_test,verbose=0)
print ('/n','----------',loss_and_metrics[1])'''