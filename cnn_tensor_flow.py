# !/usr/bin/env python

import os
import sys

import tensorflow as tf
from skimage.io import imread, imshow
import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import OneHotEncoder

import load_data as ld

fp = '/home/ubuntu/naive_bees/'
out = fp + 'CNN_tensor_flow_output.txt'

orig_stdout = sys.stdout
output = file(out, 'w')

def print_fun(string, f = output):
    '''
    write to file and screen
    '''
    print string
    f.write(string)
    f.write('\n')


# funciton from load_bees.py that handles all the loading and preprocessing
bees, Y = ld.load_bees()

train_x, test_x, train_y, test_y = train_test_split(bees, Y, test_size = 0.15,
                                                    random_state = 1234)

sess = tf.InteractiveSession()

# some conveince funcitons to initialize network weights
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

y_ = tf.placeholder("float", shape=[None, 2])
x = tf.placeholder("float", shape=[None, 48, 48, 3])

# 1st layer
W_conv1 = weight_variable([3, 3, 3, 32])
b_conv1 = bias_variable([32])

h_conv1 = tf.nn.relu(conv2d(x, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)
# second layer
W_conv2 = weight_variable([3, 3, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)
# fully connected layer, the image is 12 x 12 from two layers of max_pooling
W_fc1 = weight_variable([12 * 12 * 64, 128])
b_fc1 = bias_variable([128])

h_pool2_flat = tf.reshape(h_pool2, [-1, 12 * 12 * 64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder("float")
# dropout helps to combat overfitting
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
# output layer
W_fc2 = weight_variable([128, 2])
b_fc2 = bias_variable([2])

y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

# define loss function
cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
# training calculation
train_step = tf.train.AdamOptimizer(1e-3).minimize(cross_entropy)
# tools to evaluate the model
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
saver = tf.train.Saver()
sess.run(tf.initialize_all_variables())

# break the data into batches for training
batch_size = 200
indicies = np.array(range(train_y.shape[0]))
# this should be integer division
batch_count = train_y.shape[0]/batch_size
batches = np.array_split(indicies, batch_count)

print_fun(
    '%s batches with %s images/batch' % (str(batch_count), str(batch_size)))


# train over all batches for a certain number of epochs

# accuracy reported every 50 batches on small test set
# and every epoch on full test set
num_epochs = 25
for j in xrange(num_epochs):
    rand_ix = np.random.choice(np.array(range(test_y.shape[0])), 200)

    for i,b in enumerate(batches):
        # create batches from index
        batch_xs = train_x[b]
        batch_ys = train_y[b]
        # train CNN model
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys,
                                        keep_prob: 0.8})
        # accuracy reporting
        if i % 50 == 0:
            test_accuracy, test_loss = sess.run([accuracy, cross_entropy],
                feed_dict={x: test_x[rand_ix],
                           y_: test_y[rand_ix], keep_prob: 1.0})

            print_fun(
                "batch %d, training accuracy %s" % (i, str(
                                                        round(test_accuracy,3))))
            print_fun('total loss for batch set %s' % str(test_loss))

            yhat = sess.run( y_conv, feed_dict={x: test_x[rand_ix],
                                                y_: test_y[rand_ix],
                                                keep_prob: 1.0})
            print_fun(str(yhat.mean(0)[0]) + ', ' + str(yhat.mean(0)[1]))
            print_fun('-' * 40)

    # accuracy reporting for epoch
    print_fun('_' * 80)

    test_accuracy, test_loss = sess.run([accuracy, cross_entropy], feed_dict={
        x: test_x, y_: test_y, keep_prob: 1.0})
    # save the progress of the model
    saver.save(sess, fp + 'model.ckpt', global_step = j)
    print_fun("epoch %d, testing accuracy %s" % (j, str(round(test_accuracy,3))))
    print_fun('total loss for epoch %s' % str(test_loss))
    print_fun('_' * 80)
    print_fun('\n')


output.close()
