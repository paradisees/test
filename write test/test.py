import numpy as np
import tensorflow as tf
import csv
from sklearn.preprocessing import OneHotEncoder
from sklearn import *
INPUTNODE=25
OUTPUTNODE=2
LAYER1=300
LEARNING_RATE_BASE=0.8

data=[]
mark=[]

with open('/Users/hhy/Desktop/8.csv','r',encoding='utf-8_sig') as f1:
    csv_reader=csv.reader(f1)
    for x in csv_reader:
        data.append(list(map(int,x[0:-2])))
        mark.append(list(map(int, x[-2:])))
#Y = OneHotEncoder().fit_transform(mark).todense() #one-hot编码
X_train, X_test, y_train, y_test = cross_validation.train_test_split(
    data, mark, test_size=0.3)
def add_layer(inputs,in_size,out_size,activation_function=None):
    Weights=tf.Variable(tf.random_normal([in_size,out_size]))
    biases=tf.Variable(tf.zeros([1,out_size])+0.1)
    Wx_plus_b=tf.matmul(inputs,Weights)+biases
    if activation_function is None:
        outputs=Wx_plus_b
    else:
        outputs=activation_function(Wx_plus_b)
    return outputs

xs=tf.placeholder(tf.float32,[None,INPUTNODE])
ys=tf.placeholder(tf.float32,[None,OUTPUTNODE])

l1=add_layer(xs,INPUTNODE,LAYER1,activation_function=tf.nn.relu)
prediction=add_layer(l1,LAYER1,OUTPUTNODE,activation_function=tf.nn.softmax)

loss=tf.reduce_mean(tf.reduce_sum(tf.square(ys-prediction),reduction_indices=[1]))
train_step=tf.train.GradientDescentOptimizer(0.5).minimize(loss)
init=tf.initialize_all_variables()
sess=tf.Session()
sess.run(init)

correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(ys, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

for i in range(10000):
    sess.run(train_step,feed_dict={xs:X_train,ys:y_train})
    if i%500==0:
        print(sess.run(loss,feed_dict={xs:X_train,ys:y_train}))
acc=sess.run(accuracy,feed_dict={xs:X_test,ys:y_test})
print("准确率为你丫的：",acc)
