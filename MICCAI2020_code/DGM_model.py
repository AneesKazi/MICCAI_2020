import tensorflow as tf
import math
import time
import numpy as np
import os
import sys
import tf_util

from layers import DGM, EdgeConv


def placeholder_inputs(batch_size, num_point, num_features, num_classes):
    pointclouds_pl = tf.compat.v1.placeholder(tf.float32, shape=(batch_size, num_point, num_features))
    labels_pl = tf.compat.v1.placeholder(tf.int32, shape=(batch_size, num_point, num_classes))

    return pointclouds_pl, labels_pl


def ADGM(features, is_training, fc_layers=[8, 8], scope=''):  # tadpole: [32,16]
    net = features
    print(net.shape)
    for i, fc in enumerate(fc_layers):
        net = tf_util.conv2d(net, fc, [1, 1], padding='VALID', stride=[1, 1],
                             bn=False, is_training=is_training, scope='adgm%d' % i, is_dist=True)
        # print(net.shape)

        # net = tf.nn.dropout(net,keep_prob=1-0.2*tf.cast(is_training,'float'))

        # part 1: Graph representation feature learning
    X_hat = net

    # Probabilistic graph generator
    A = -tf_util.pairwise_distance(X_hat)
    print('here:', 1 - tf.eye(A.get_shape().as_list()[-1]))

    temp = (1 + tf_util._variable_with_weight_decay(scope + 'temp', [1], 1e-1, None, use_xavier=False))
    # th = (5 + tf_util._variable_with_weight_decay(scope+'th', [X_hat.shape[1],1], 1e-6, None, use_xavier=False))
    th = (5 + tf_util._variable_with_weight_decay(scope + 'th', [1, 1], 1e-1, None, use_xavier=False))
    print('temp:',temp)
    print('theta:',th)

    A = tf.nn.sigmoid(temp * A + th)
    A = A * (1 - tf.eye(A.get_shape().as_list()[-1])) + tf.eye(A.get_shape().as_list()[-1])
    A = A / tf.reduce_sum(A, -1)[..., None]
    return A, th


def get_model(input_features, num_classes, is_training):
    fc_layers = [8, 8]

    # input shapet
    net = tf.expand_dims(input_features, -2)

    A, th = ADGM(net, is_training)
    #   A = tf.identity(A,name='aff')
    # #   A = tf.eye(564,564)[None,:]
    print(A.shape)

    net = tf.matmul(A[0], net[0, :, 0, :])
    net = tf.expand_dims(tf.expand_dims(net, axis=0), axis=2)
    print(net.shape)
    net = tf_util.conv2d(net, 8, [1, 1], padding='VALID', stride=[1, 1],
                         bn=False, is_training=is_training, scope='GC', is_dist=True)
    print(net.shape)

    net = tf.nn.relu(net)
    for i, fc in enumerate(fc_layers):
        net = tf_util.conv2d(net, fc, [1, 1], padding='VALID', stride=[1, 1],
                             bn=False, is_training=is_training, scope='dp%d' % i, is_dist=True)
        # net = tf_util.dropout(net, keep_prob=0.6, is_training=is_training, scope='dp%d'%i)

    net = tf_util.conv2d(net, num_classes, [1, 1], padding='VALID', stride=[1, 1],
                         activation_fn=None, scope='seg/conv3%d' % i, is_dist=True)
    net = tf.squeeze(net, [2])
    print(net.shape)

    return net, A, th


def masked_softmax_cross_entropy(preds, labels, mask, num_classes):
    """Softmax cross-entropy loss with masking."""
    loss = tf.nn.softmax_cross_entropy_with_logits(logits=preds, labels=labels)
    mask = tf.cast(mask, dtype=tf.float32)
    mask /= tf.reduce_mean(mask)

    print(mask.shape)

    loss *= mask

    return loss


def get_loss(pred, label, mask):
    """ pred: B,N,13; label: B,N """
    num_classes = pred.shape[-1]
    loss = masked_softmax_cross_entropy(pred, label, mask, num_classes)

    return tf.reduce_mean(loss)
