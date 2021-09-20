from __future__ import division
import os
import time
import tensorflow as tf
import numpy as np
import pickle
import random
import sys
from random import choice
import skimage
from skimage import morphology
from layers import conv2d

class HCR(object):
    def __init__(self,
                 num_classes=1,
                 keep_prob=0.5,
                 feature_root=32,
                 ksize=3,
                 en_block_nums=5,
                 lm_blocks_nums=2,
                 gm_blocks_nums=2,
                 sinusoid_table=np.array([i for i in range(100)]),
                 net_name='HCR',
                 reuse=False,
                 is_activation=True,
                 is_training = True,
                 activation_fn="relu"):
        """
        Implements HCR architecture
        :param num_classes: Number of output classes.
        :param keep_prob: Dropout keep probability, set to 0.0 if not training or if no dropout is desired.
        :param feature_root: The number of output channels in the first level, this will be doubled every level.
        :param ksize: convolution size
        :en_block_nums: the number of convolutional block in encoder path of feature extraction module
        :de_block_nums: the number of convolutional block in decoder path of feature extraction module
        :lm_blocks_nums: the number of local modelling block
        :param net_name: the number of global modelling block
        :sinusoid_table: position encoding
        :param is_activation:
        :param activation_fn: The activation function.
        """
        self.num_classes = num_classes
        self.keep_prob = keep_prob
        self.feature_root = feature_root
        self.ksize=ksize
        self.sinusoid_table = sinusoid_table,
        self.en_block_nums=en_block_nums
        self.lm_blocks_nums = lm_blocks_nums
        self.gm_blocks_nums = gm_blocks_nums
        self.net_name=net_name
        self.reuse=reuse
        self.is_training = is_training
        self.is_activation = is_activation
        self.activation_fn = activation_fn

    def global_modelling_fn(self, x, sinusoid_table, name, input_channels, output_channels, ksizes=[1, 8, 8, 1], strides=[1, 8, 8, 1]):
        shapes=tf.shape(x)
        q, w, b = conv2d(x, [1, 1, input_channels, output_channels], '%s_q' % (name))
        k, w, b = conv2d(x, [1, 1, input_channels, output_channels], '%s_k' % (name))
        v, w, b = conv2d(x, [1, 1, input_channels, output_channels], '%s_v' % (name))
        q = tf.reshape(tf.extract_image_patches(q, ksizes=ksizes, strides=strides, rates=[1, 1, 1, 1],padding="SAME"), (-1, (shapes[1]//strides[1])**2, ksizes[1]*ksizes[1]*output_channels)) + sinusoid_table
        k = tf.reshape(tf.extract_image_patches(k, ksizes=ksizes, strides=strides, rates=[1, 1, 1, 1],padding="SAME"), (-1, (shapes[1]//strides[1])**2, ksizes[1]*ksizes[1]*output_channels)) + sinusoid_table
        k = tf.transpose(k, (0, 2, 1))
        v = tf.reshape(tf.extract_image_patches(v, ksizes=ksizes, strides=strides, rates=[1, 1, 1, 1],padding="SAME"), (-1, (shapes[1]//strides[1])**2, ksizes[1]*ksizes[1]*output_channels))
        qk = tf.matmul(q, k) / tf.sqrt(ksizes[1]*ksizes[1]*output_channels)
        qk = tf.nn.softmax(qk, axis=1)
        transformer = tf.matmul(qk, v)
        transformer = tf.transpose(tf.reshape(transformer, (-1, shapes[1]//strides[1], shapes[1]//strides[1], ksizes[1], ksizes[1], output_channels)), (0, 1, 3, 2, 4, 5))
        transformer = tf.reshape(transformer, (-1, shapes[1],  shapes[2], output_channels))
        transformer = tf.nn.relu(tf.contrib.layers.layer_norm(transformer))
        return transformer

    def HCR_NET(self,x):
        x_input_shapes = int(x.get_shape()[-1])

        features = list()
        weights = list()
        bias= list()

        #-----begin feature extraction-----
        with tf.variable_scope('%s_en' % (self.name), reuse=self.reuse) as scope:
            #encoder
            for i in range(self.en_block_nums):
                input_channels=   2 ** (i-1) * self.feature_root
                middle_channels = 2 ** i * self.feature_root
                output_channels = 2 ** i * self.feature_root

                if i==0:
                    input_channels=x_input_shapes
                if i==self.en_block_nums-1:
                    output_channels = 2 ** (i-1) * self.feature_root

                x, w, b = conv2d(x, [self.ksize, self.ksize, input_channels, middle_channels],'en%d_1' % (i))
                weights.append(w)
                bias.append(b)
                x = tf.nn.dropout(x, self.keep_prob)
                x, w, b = conv2d(x, [self.ksize, self.ksize, middle_channels, output_channels],'en%d_2' % (i))
                weights.append(w)
                bias.append(b)

                if i !=self.en_block_nums-1:
                    features.append(x)
                    x = tf.nn.max_pool(x, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')

        shapes = tf.shape(x)
        with tf.variable_scope('%s_de' % (self.name), reuse=self.reuse) as scope:
            # decoder
            for i in range(self.en_block_nums-1, 0, -1):
                input_channels = 2 ** i * self.feature_root
                middle_channels = 2 ** (i-1) * self.feature_root
                output_channels = 2 ** (i-2) * self.feature_root

                if i == 1:
                    output_channels = 2 ** (i - 1) * self.feature_root

                up_factor=2**(self.en_block_nums-i)
                up=tf.image.resize_images(x, [up_factor*shapes[1], up_factor*shapes[1]], method=tf.image.ResizeMethod.BILINEAR)
                fmerge=tf.concat([up,features[i-self.en_block_nums]],axis=-1)
                x, w, b = conv2d(fmerge, [self.ksize, self.ksize, input_channels, middle_channels],'de%d_1' % (i))
                weights.append(w)
                bias.append(b)
                x, w, b = conv2d(x, [self.ksize, self.ksize, middle_channels, output_channels],'de%d_2' % (i))
                weights.append(w)
                bias.append(b)
        # -----end feature extraction-----

        # -----begin local_modelling-----
        with tf.variable_scope('%s_local_modelling' % (self.name), reuse=self.reuse) as scope:
            lx=tf.nn.dropout(x,self.keep_prob)
            for i in range(self.lm_blocks_nums):
                lx, w, b = conv2d(lx, [self.ksize, self.ksize, middle_channels, output_channels], 'lm_%d' % (i))
            logits_lm, w, b = conv2d(lx, [1, 1, self.feature_root, self.num_classes], 'logits_lm', is_activation=False, is_BN=False)
            output_lm = tf.sigmoid(logits_lm)
        # -----end local_modelling-----

        # -----begin global_modelling-----
        with tf.variable_scope('%s_global_modelling' % (self.name), reuse=self.reuse) as scope:
            gx = tf.nn.dropout(x, self.keep_prob)
            for i in range(self.gm_blocks_nums):
                gx= self.global_modelling_fn(x=gx, sinusoid_table=self.sinusoid_table, name='gm_%d'%(i), input_channels=self.feature_root, output_channels=self.feature_root)
            logits_gm, w, b = conv2d(gx, [1, 1, self.feature_root, self.num_classes], 'logits_gm',is_activation=False, is_BN=False)
            output_gm = tf.sigmoid(logits_gm)
        # -----begin global_modelling-----
        return output_lm,output_gm,weights,bias



















