import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import csv
from sklearn import cross_validation
INPUT_NODE = 100  # 输入节点
OUTPUT_NODE = 2  # 输出节点
LAYER1_NODE = 500  # 隐藏层数
BATCH_SIZE = 100  # 每次batch打包的样本个数
# 模型相关的参数
LEARNING_RATE_BASE = 0.8
LEARNING_RATE_DECAY = 0.99
REGULARAZTION_RATE = 0.0001
TRAINING_STEPS = 30000
MOVING_AVERAGE_DECAY = 0.99

train_data=[]
train_mark=[]
test_data=[]
test_mark=[]
data=[]
mark=[]
'''
with open('/Users/hhy/Desktop/vectortrain.csv','r',encoding='utf-8_sig') as f:
    csv_reader=csv.reader(f)
    for x in csv_reader:
        data.append(list(map(int,x[0:-2])))
        mark.append(list(map(int, x[-2:])))
train_data, train_mark, test_data, test_mark = cross_validation.train_test_split(
    data, mark, test_size=0.3,random_state=1)
'''
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

def inference(input_tensor, avg_class, weights1, biases1, weights2, biases2):
    # 不使用滑动平均类
    if avg_class == None:
        layer1 = tf.nn.relu(tf.matmul(input_tensor, weights1) + biases1)
        return tf.matmul(layer1, weights2) + biases2

    else:
        # 使用滑动平均类
        layer1 = tf.nn.relu(tf.matmul(input_tensor, avg_class.average(weights1)) + avg_class.average(biases1))
        return tf.matmul(layer1, avg_class.average(weights2)) + avg_class.average(biases2)


x = tf.placeholder(tf.float32, [None, INPUT_NODE], name='x-input')
y_ = tf.placeholder(tf.float32, [None, OUTPUT_NODE], name='y-input')
# 生成隐藏层的参数。
weights1 = tf.Variable(tf.truncated_normal([INPUT_NODE, LAYER1_NODE], stddev=0.1))
biases1 = tf.Variable(tf.constant(0.1, shape=[LAYER1_NODE]))
# 生成输出层的参数。
weights2 = tf.Variable(tf.truncated_normal([LAYER1_NODE, OUTPUT_NODE], stddev=0.1))
biases2 = tf.Variable(tf.constant(0.1, shape=[OUTPUT_NODE]))

# 计算不含滑动平均类的前向传播结果
y = inference(x, None, weights1, biases1, weights2, biases2)

# 定义训练轮数及相关的滑动平均类
global_step = tf.Variable(0, trainable=False)
variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
variables_averages_op = variable_averages.apply(tf.trainable_variables())
average_y = inference(x, variable_averages, weights1, biases1, weights2, biases2)

# 计算交叉熵及其平均值
cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
cross_entropy_mean = tf.reduce_mean(cross_entropy)

# 损失函数的计算
regularizer = tf.contrib.layers.l2_regularizer(REGULARAZTION_RATE)
regularaztion = regularizer(weights1) + regularizer(weights2)
loss = cross_entropy_mean + regularaztion

# 设置指数衰减的学习率。
learning_rate = tf.train.exponential_decay(
    LEARNING_RATE_BASE,
    global_step,
    500,
    LEARNING_RATE_DECAY,
    staircase=True)

# 优化损失函数
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

# 反向传播更新参数和更新每一个参数的滑动平均值
with tf.control_dependencies([train_step, variables_averages_op]):
    train_op = tf.no_op(name='train')

# 计算正确率
correct_prediction = tf.equal(tf.argmax(average_y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# 初始化会话，并开始训练过程。
with tf.Session() as sess:
    tf.global_variables_initializer().run()
    train_feed = {x: train_data, y_: train_mark}
    test_feed = {x: test_data, y_: test_mark}

    # 循环的训练神经网络。
    for i in range(TRAINING_STEPS):
        if i % 1000 == 0:
            acc = sess.run(accuracy, feed_dict=test_feed)
            print("准确率为 " ,acc)
        sess.run(train_op, feed_dict={x: train_data, y_: train_mark})

    test_acc = sess.run(accuracy, feed_dict=test_feed)
    print("After %d training step(s), test accuracy using average model is %g" % (TRAINING_STEPS, test_acc))




