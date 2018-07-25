import numpy as np
import tensorflow as tf
from tensorflow.contrib.learn.python.learn.datasets import base
INPUTNODE=100
OUTPUTNODE=2
LAYER1=10
filenametrain = "/Users/hhy/Desktop/vectortrain.csv"
train = base.load_csv_without_header(filename=filenametrain, target_dtype=np.int, features_dtype=np.int)
filenametest = "/Users/hhy/Desktop/vectortest.csv"
test = base.load_csv_without_header(filename=filenametest, target_dtype=np.int, features_dtype=np.int)

def add_layer(inputs,in_size,out_size,activation_function=None):
    Weights=tf.Variable(tf.random_normal([in_size,out_size]))
    biases=tf.Variable(tf.zeros([1,out_size])+0.1)
    Wx_plus_b=tf.matmul(inputs,Weights)+biases
    if activation_function is None:
        outputs=Wx_plus_b
    else:
        outputs=activation_function(Wx_plus_b)
    return outputs
x_train=train.data.reshape(24780,100)
#y_train=train.target.reshape(24780,1)
y_train=np.array(train.target).reshape(24780,2)
#y_train=tf.one_hot(y_train1,2,1,0)

x_test=test.data.reshape(754,100)
y_test=np.array(test.target).reshape(754,2)
#y_test=test.target.reshape(754,1)
#y_test=tf.one_hot(y_test1,2,1,0)

xs=tf.placeholder(tf.float32,[None,INPUTNODE])
ys=tf.placeholder(tf.float32,[None,OUTPUTNODE])

l1=add_layer(xs,INPUTNODE,LAYER1,activation_function=tf.nn.relu)
prediction=add_layer(l1,LAYER1,OUTPUTNODE,activation_function=tf.nn.softmax)

loss=tf.reduce_mean(tf.reduce_sum(tf.square(ys-prediction),reduction_indices=[1]))
train_step=tf.train.GradientDescentOptimizer(0.1).minimize(loss)
init=tf.initialize_all_variables()
sess=tf.Session()
sess.run(init)

correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(ys, 1))
#print(tf.argmax(prediction,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

for i in range(1000):
    sess.run(train_step,feed_dict={xs:x_train,ys:y_train})
    if i%50==0:
        print(sess.run(loss,feed_dict={xs:x_train,ys:y_train}))
acc=sess.run(accuracy,feed_dict={xs:x_test,ys:y_test})
print(acc)
