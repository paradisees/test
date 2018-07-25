from sklearn.preprocessing import OneHotEncoder
from sklearn import cross_validation
'''Y_data=[[0],[1],[0],[1]]
Y = OneHotEncoder().fit_transform(Y_data).todense() #one-hot编码
print(Y)'''
import numpy as np
import tensorflow as tf
import csv
INPUTNODE=100
OUTPUTNODE=2
LAYER1=1000
LEARNING_RATE_BASE=0.8

data=[]
mark=[]
with open('/Users/hhy/Desktop/Data.csv','r',encoding='utf-8_sig') as f:
    csv_reader=csv.reader(f)
    for x in csv_reader:
        data.append(list(map(float,x[0:-1])))
        mark.append([int(x[-1])])
Y = OneHotEncoder().fit_transform(mark).todense().astype(int) #one-hot编码
train_data, test_data, train_mark, test_mark = cross_validation.train_test_split(
data, Y, test_size=0.1)
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

for i in range(5000):
    sess.run(train_step,feed_dict={xs:train_data,ys:train_mark})
    if i%500==0:
        print(sess.run(loss,feed_dict={xs:train_data,ys:train_mark}))
acc=sess.run(accuracy,feed_dict={xs:test_data,ys:test_mark})
print("准确率为你丫的：",acc)
# 0.669628