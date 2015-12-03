#!/usr/bin/env python

import os
import sys

import tensorflow as tf
from skimage.io import imread, imshow
import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import OneHotEncoder

import load_data as ld

out = "/home/ubuntu/CNN_tensor_flow_output.txt"



orig_stdout = sys.stdout
f = file(out, 'w')
sys.stdout = f

bees, Y = ld.load_bees()

train_x, test_x, train_y, test_y = train_test_split(bees, Y, test_size = 0.15)

sess = tf.InteractiveSession()

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

W_conv1 = weight_variable([8, 8, 3, 32])
b_conv1 = bias_variable([32])

x = tf.placeholder("float", shape=[None, 48, 48, 3])
y_ = tf.placeholder("float", shape=[None, 2])

x_image = tf.reshape(x, [-1, 48, 48, 3])

h_conv1 = tf.clip_by_norm(tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1), 10)
h_pool1 = max_pool_2x2(h_conv1)

W_conv2 = weight_variable([8, 8, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.clip_by_norm(tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2), 10)
h_pool2 = max_pool_2x2(h_conv2)

W_fc1 = weight_variable([50 * 50 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 50*50*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([1024, 2])
b_fc2 = bias_variable([2])

y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

# I'm adding all the loss functions to this collection
# adding l2 regularization to try to keep the weights
# in the relu layers < inf
losses = tf.get_collection('losses')
tf.add_to_collection('losses', tf.nn.l2_loss(h_conv1))
tf.add_to_collection('losses', tf.nn.l2_loss(h_conv2))
tf.add_to_collection('losses', -tf.reduce_sum(y_*tf.log(y_conv)))

loss = tf.add_n(tf.get_collection('losses'), name = 'total_loss')

# cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)
# clipping_1 = tf.clip_by_norm(h_conv1, 1)
# clipping_2 = tf.clip_by_norm(h_conv2, 1)
# trying a different optimization algorithm
# train_step = tf.train.GradientDescentOptimizer(
#                         learning_rate = 0.01).minimize(loss)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
# object to save TF session
saver = tf.train.Saver()
sess.run(tf.initialize_all_variables())

for i in range(0,len(train_y),10):
    end = i + 10
    batch_xs = train_x[i:end]
    batch_ys = train_y[i:end]
    if i%100 == 0:
        train_accuracy = accuracy.eval(feed_dict={
            x: batch_xs, y_: batch_ys, keep_prob: 1.0})
        saver.save(sess, fp + '/model.ckpt', global_step = i)
        print "step %d, training accuracy %s"%(i, str(round(train_accuracy,3)))
        print 'total loss %s' % str(loss)
    train_step.run(feed_dict={x: batch_xs, y_: batch_ys, keep_prob: 0.5})

feed_dict_3 = {x: test_x[:10], keep_prob: 1.0}
classification = sess.run(y_conv, feed_dict_3)

print classification



sys.stdout = orig_stdout
f.close()
