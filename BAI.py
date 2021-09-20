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

class BAI(object):
    def __init__(self,
                 num_classes=1,
                 keep_prob=1.0,
                 feature_root=32,
                 ksize=3,
                 en_block_nums=5,
                 net_name='BAI',
                 reuse=False,
                 is_activation=True,
                 is_training = True,
                 activation_fn="relu"):
        """
        Implements BAI architecture
        :param num_classes: Number of output classes.
        :param keep_prob: Dropout keep probability, set to 0.0 if not training or if no dropout is desired.
        :param feature_root: The number of output channels in the first level, this will be doubled every level.
        :param ksize: convolution size
        :en_block_nums: the number of convolutional block in encoder path of feature extraction module
        :de_block_nums: the number of convolutional block in decoder path of feature extraction module
        :param net_name: the number of global modelling block
        :param is_activation:
        :param activation_fn: The activation function.
        """
        self.num_classes = num_classes
        self.keep_prob = keep_prob
        self.feature_root = feature_root
        self.ksize=ksize
        self.en_block_nums=en_block_nums
        self.net_name=net_name
        self.reuse=reuse
        self.is_training = is_training
        self.is_activation = is_activation
        self.activation_fn = activation_fn

    def BAI_NET(self,x):
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
                x = tf.nn.dropout(x, self.keep_prob)
                x, w, b = conv2d(x, [self.ksize, self.ksize, middle_channels, output_channels],'en%d_2' % (i))

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
                x, w, b = conv2d(x, [self.ksize, self.ksize, middle_channels, output_channels],'de%d_2' % (i))
            logits, w, b = conv2d(x, [1, 1, self.feature_root, self.num_classes], 'logits', is_activation=False, is_BN=False)
            output = tf.sigmoid(logits)
        return logits, output



















