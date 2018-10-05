import numpy as np
import tensorflow as tf

def conv2d(layer_id, filter_num, x):
    w = tf.get_variable("w%d"%layer_id, shape=[3,3,x.shape[3],filter_num], 
            initializer=tf.contrib.layers.xavier_initializer())
    conv = tf.nn.conv2d(x, w, (1,1,1,1), 'SAME')
    b = tf.get_variable('b%d'%layer_id, shape=[filter_num], 
            initializer=tf.constant_initializer(0))
    relu = tf.nn.relu(conv + b)
    return relu

def mnist_model(input_shape, filters, classes):
    x = tf.placeholder(
            shape=[None,input_shape[0],input_shape[1],input_shape[2]], 
            dtype=tf.float32)
    for i in range(len(filters)):
        if i==0:
            y = conv2d(i, filters[i], x)
        else:
            # down sampling using subpixel convolution
            y = tf.space_to_depth(y, 2)
            y = conv2d(i, filters[i], y)
    y = tf.reshape(y, [-1, y.shape[1]*y.shape[2]*y.shape[3]])
    y = tf.layers.dense(y, classes, activation=None, use_bias=True, 
            kernel_initializer=tf.truncated_normal_initializer, 
            bias_initializer=tf.zeros_initializer())
    y = tf.nn.softmax(y)
    labels = tf.argmax(y)
    gt_labels = tf.placeholder(shape=[None], dtype=tf.int32)
    one_hot_y = tf.one_hot(gt_labels, classes, 1.0, 0.0)
    err = tf.reduce_mean(tf.square(y-one_hot_y))
    return (x, labels, gt_labels, err)
