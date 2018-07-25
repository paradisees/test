import tensorflow as tf
import csv
import numpy as np
INPUT_NODE = 100  # 输入节点
OUTPUT_NODE = 2  # 输出节点

NUM_CHANNELS = 1
NUM_LABELS = 2

CONV1_DEEP = 32
CONV1_SIZE = 5

CONV2_DEEP = 64
CONV2_SIZE = 5

FC_SIZE = 512
LEARNING_RATE_BASE = 0.8
LEARNING_RATE_DECAY = 0.99
REGULARAZTION_RATE = 0.0001
TRAINING_STEPS = 30000
MOVING_AVERAGE_DECAY = 0.99
train_data=[]
train_mark=[]
test_data=[]
test_mark=[]
with open('/Users/hhy/Desktop/1.csv','r',encoding='utf-8_sig') as f1:
    csv_reader=csv.reader(f1)
    for x in csv_reader:
        train_data.append(list(map(int,x[0:-2])))
        train_mark.append(list(map(int, x[-2:])))
with open('/Users/hhy/Desktop/2.csv','r',encoding='utf-8_sig') as f2:
    csv_reader=csv.reader(f2)
    for x in csv_reader:
        test_data.append(list(map(int,x[0:-2])))
        test_mark.append(list(map(int, x[-2:])))

def inference(input_tensor, regularizer):
    with tf.variable_scope('layer1-conv1'):
        conv1_weights = tf.get_variable(
            "weight", [CONV1_SIZE, CONV1_SIZE, NUM_CHANNELS, CONV1_DEEP],
            initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv1_biases = tf.get_variable("bias", [CONV1_DEEP], initializer=tf.constant_initializer(0.0))
        conv1 = tf.nn.conv2d(input_tensor, conv1_weights, strides=[1, 1, 1, 1], padding='SAME')
        relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_biases))

    with tf.name_scope("layer2-pool1"):
        pool1 = tf.nn.max_pool(relu1, ksize = [1,2,2,1],strides=[1,2,2,1],padding="SAME")

    with tf.variable_scope("layer3-conv2"):
        conv2_weights = tf.get_variable(
            "weight", [CONV2_SIZE, CONV2_SIZE, CONV1_DEEP, CONV2_DEEP],
            initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv2_biases = tf.get_variable("bias", [CONV2_DEEP], initializer=tf.constant_initializer(0.0))
        conv2 = tf.nn.conv2d(pool1, conv2_weights, strides=[1, 1, 1, 1], padding='SAME')
        relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_biases))

    with tf.name_scope("layer4-pool2"):
        pool2 = tf.nn.max_pool(relu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        pool_shape = pool2.get_shape().as_list()
        nodes = pool_shape[1] * pool_shape[2] * pool_shape[3]
        reshaped = tf.reshape(pool2, [pool_shape[0], nodes])

    with tf.variable_scope('layer5-fc1'):
        fc1_weights = tf.get_variable("weight", [nodes, FC_SIZE],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
        if regularizer != None: tf.add_to_collection('losses', regularizer(fc1_weights))
        fc1_biases = tf.get_variable("bias", [FC_SIZE], initializer=tf.constant_initializer(0.1))

        fc1 = tf.nn.relu(tf.matmul(reshaped, fc1_weights) + fc1_biases)
        fc1 = tf.nn.dropout(fc1, 0.5)

    with tf.variable_scope('layer6-fc2'):
        fc2_weights = tf.get_variable("weight", [FC_SIZE, NUM_LABELS],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
        if regularizer != None: tf.add_to_collection('losses', regularizer(fc2_weights))
        fc2_biases = tf.get_variable("bias", [NUM_LABELS], initializer=tf.constant_initializer(0.1))
        logit = tf.matmul(fc1, fc2_weights) + fc2_biases

    return logit

train_data,test_data=np.array(train_data).astype(np.float32),np.array(test_data).astype(np.float32)
x = tf.placeholder(tf.float32, [None,100,1,1], name='x-input')
y_ = tf.placeholder(tf.float32, [None, OUTPUT_NODE], name='y-input')
regularizer = tf.contrib.layers.l2_regularizer(REGULARAZTION_RATE)
'''batch_size = 8 # 使用MBGD算法，设定batch_size为8
def generatebatch(X,Y,n_examples, batch_size):
    for batch_i in range(n_examples // batch_size):
        start = batch_i*batch_size
        end = start + batch_size
        batch_xs = X[start:end]
        batch_ys = Y[start:end]
        yield batch_xs, batch_ys # 生成每一个batch'''
xs=train_data.reshape(-1,100,1,1)
xs1=test_data.reshape(-1,100,1,1)
#xs=tf.cast(x1s, tf.float32)
#xs1=tf.cast(x1s1, tf.float32)
#tf.reset_default_graph()
y=inference(x, regularizer)
global_step=tf.Variable(0,trainable=False)
variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
variables_averages_op = variable_averages.apply(tf.trainable_variables())
cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
cross_entropy_mean = tf.reduce_mean(cross_entropy)
loss = cross_entropy_mean + tf .add_n(tf.get_collection('losses'))
learning_rate = tf.train.exponential_decay(
    LEARNING_RATE_BASE,
    global_step,
    500,
    LEARNING_RATE_DECAY,
    staircase=True)
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
with tf.control_dependencies([train_step,variables_averages_op]):
    train_op=tf.no_op(name='train')
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
# 初始化会话，并开始训练过程。

with tf.Session() as sess:
    tf.global_variables_initializer().run()
    train_feed = {x: xs, y_: train_mark}
    test_feed = {x: xs1, y_: test_mark}
    # 循环的训练神经网络。
    for i in range(TRAINING_STEPS):
        if i % 10 == 0:
            acc = sess.run(accuracy, feed_dict=test_feed)
            print("准确率为 " ,acc)
        sess.run(train_op, feed_dict=train_feed)

    #test_acc = sess.run(accuracy, feed_dict=test_feed)
    #print("After %d training step(s), test accuracy using average model is %g" % (TRAINING_STEPS, test_acc))


'''
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(TRAINING_STEPS):
        for batch_xs,batch_ys in generatebatch(xs,train_mark,len(train_mark),batch_size): # 每个周期进行MBGD算法
            sess.run(train_step,feed_dict={x:batch_xs,y_:batch_ys})
        if(epoch%1000==0):
            res = sess.run(accuracy,feed_dict={x:xs1,y_:test_mark})
            print (epoch,res)
    #res_ypred = y_pred.eval(feed_dict={tf_X:X,tf_Y:Y}).flatten() # 只能预测一批样本，不能预测一个样本
    #print (res_ypred)'''