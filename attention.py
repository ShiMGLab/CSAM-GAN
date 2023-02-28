import tensorflow as tf
from keras.layers import Dropout
from keras.layers import Input, Dense, Lambda, Layer, Add, BatchNormalization, Dropout, Activation, merge,  \
    MaxPooling2D, Activation, LeakyReLU, concatenate
import numpy as np
from keras.utils.generic_utils import get_custom_objects

def cbam(inputs):


    # channel attention
    maxpool_channel = tf.reduce_max(inputs, axis=1, keepdims=True)
    avgpool_channel = tf.reduce_mean(inputs, axis=1, keepdims=True)

    mlp_1_max = Dense(128,activation='relu')(maxpool_channel)
    mlp_2_max = Dense(256)(mlp_1_max)

    mlp_1_avg = Dense(128,activation='relu')(avgpool_channel)
    mlp_2_avg = Dense(256)(mlp_1_avg)

    channel_attention = tf.nn.sigmoid(mlp_2_max + mlp_2_avg)
    channel_refined_feature = inputs * channel_attention

    # spatial attention neuron attention
    maxpool_spatial = tf.reduce_max(channel_refined_feature, axis=1, keepdims=True)
    avgpool_spatial = tf.reduce_mean(channel_refined_feature, axis=1, keepdims=True)
    max_avg_pool_spatial = tf.concat([maxpool_spatial, avgpool_spatial], axis=1)
    conv=Dense(256)(max_avg_pool_spatial)
    #conv_layer = Dense(256)(conv)
    spatial_attention = tf.nn.sigmoid(conv)
    refined_feature = channel_refined_feature * spatial_attention

    return refined_feature




