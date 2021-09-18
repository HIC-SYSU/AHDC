from __future__ import division
import tensorflow as tf
import numpy as np
import random
import sys
from random import choice

def conv2d(inputs, filters, name, stridess=[1, 1, 1, 1], is_activation=True, activation_fn='relu',is_BN=True, is_training=True):
    w = tf.get_variable('w_%s' % (name), filters, initializer=tf.truncated_normal_initializer(stddev=0.1))
    b = tf.get_variable('b_%s' % (name), filters[3], initializer=tf.constant_initializer(0.01))
    conv = tf.nn.conv2d(inputs, w, strides=stridess, padding='SAME')+b
    if is_BN == True:
        conv = tf.layers.batch_normalization(conv, training=is_training)
    if is_activation == True:
        if activation_fn == 'relu':
            conv = tf.nn.relu(conv)
        if activation_fn == 'sigmoid':
            conv = tf.sigmoid(conv)
    return conv, w, b

# parametric leaky relu
def prelu(x):
	alpha = tf.get_variable('alpha', shape=x.get_shape()[-1], dtype=x.dtype, initializer=tf.constant_initializer(0.1))
	return tf.maximum(0.0, x) + alpha * tf.minimum(0.0, x)

def get_position_angle_vec(position,d_hid):
    return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

def PositionalEncoding(n_position, d_hid):
    sinusoid_table = np.array([get_position_angle_vec(pos_i,d_hid) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1
    return sinusoid_table

def loss_function(logits, labels, loss_type='dice'):
    smooth = 0.0000001
    loss_value = 0
    if loss_type == 'dice':
        inse = tf.reduce_sum(logits * labels, axis=[1, 2, 3], keep_dims=True)
        p = tf.reduce_sum(logits, axis=[1, 2, 3], keep_dims=True)
        g = tf.reduce_sum(labels, axis=[1, 2, 3], keep_dims=True)
        loss_value = tf.reduce_mean(1 - (2.0 * inse + smooth) / (p + g + smooth))
    if loss_type == 'mse':
        loss_value = tf.reduce_mean(tf.reduce_mean(tf.square(logits - labels),axis=[1,2,3]))
        # loss_value = tf.reduce_mean(tf.reduce_mean(tf.abs(logits - labels),axis=[1,2,3]))
    if loss_type == 'sigmoid_cross_entrop':
        loss_value = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels))
    return loss_value

def sigmoid_rampup(current, rampup_length):
    """Exponential rampup from https://arxiv.org/abs/1610.02242"""
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return np.exp(-5.0 * phase * phase)*1.0

