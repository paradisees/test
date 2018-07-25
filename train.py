import tensorflow as tf
import csv
import inference
import os
LEARNING_RATE_BASE = 0.8
LEARNING_RATE_DECAY = 0.99
REGULARAZTION_RATE = 0.0001
TRAINING_STEPS = 30000
MOVING_AVERAGE_DECAY = 0.99
MODEL_SAVE_PATH='/Users/hhy/Desktop/'
MODEL_NAME="model.ckpt"
BATCH_SIZE=500
train_data=[]
train_mark=[]
test_data=[]
test_mark=[]
data=[]
mark=[]
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
def train(mnist):
    x = tf.placeholder(tf.float32, [None, inference.INPUT_NODE], name='x-input')
    y_ = tf.placeholder(tf.float32, [None, inference.OUTPUT_NODE], name='y-input')
    regularizer = tf.contrib.layers.l2_regularizer(REGULARAZTION_RATE)
    y=inference.inference(x, regularizer)
    global_step=tf.Variable(0,trainable=False)
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))
    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE,
        global_step,
        mnist.train.num_examples/BATCH_SIZE,
        LEARNING_RATE_DECAY,
        staircase=True)
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
    with tf.control_dependencies([train_step,variables_averages_op]):
        train_op=tf.no_op(name='train')
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    saver = tf.train.Saver()
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        #train_feed = {x: train_data, y_: train_mark}
        #test_feed = {x: test_data, y_: test_mark}

        for i in range(TRAINING_STEPS):
            xs,ys=mnist.train.next_batch(BATCH_SIZE)
            _,loss_value,step=sess.run([train_op,loss,global_step],feed_dict={x:xs,y_:ys})
            if i%1000==0:
                print("after %d training,loss is %g" % (step,loss_value))
            saver.save(sess,os.path.join(MODEL_SAVE_PATH,MODEL_NAME),global_step=global_step)
        '''
        for i in range(TRAINING_STEPS):
            if i % 1000 == 0:
                acc = sess.run(accuracy, feed_dict=test_feed)
                print("准确率为 ", acc)
            sess.run(train_op, feed_dict={x: train_data, y_: train_mark})

        test_acc = sess.run(accuracy, feed_dict=test_feed)
        print("After %d training step(s), test accuracy using average model is %g" % (TRAINING_STEPS, test_acc))'''
train(mnist)

