import tensorflow as tf
import math
import time
import numpy as np
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
sys.path.append(os.path.join(BASE_DIR, '../models'))
import tf_util


def placeholder_inputs(batch_size, num_point, num_features, num_classes):
    pointclouds_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point, num_features))
    labels_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point, num_classes))
    # labels_pl = tf.placeholder(tf.int32, shape=(batch_size, num_point, num_classes))
    return pointclouds_pl, labels_pl

def dot(x, y, sparse=False):
    """Wrapper for tf.matmul (sparse vs dense)."""
    if sparse:
        res = tf.sparse_tensor_dense_matmul(x, y)
    else:
        res = tf.matmul(x, y)
    return res


def dgcnn_layer(data, knnchs, outchs=[64, 64], k=20, scope='', is_training=True, bn_decay=0.9, weight_decay=0,
                nn_idx=None, knn_data=None, P=None):
    if knn_data is None:
        knn_data = data

    adj = P
    if adj is None:
        X = tf_util.conv2d(knn_data, knnchs * 2, [1, 1],
                           padding='VALID', stride=[1, 1],
                           bn=False, is_training=is_training,
                           scope=scope + 'graph_mlp_ml1', bn_decay=bn_decay, is_dist=True)

        X = tf_util.conv2d(X, knnchs, [1, 1],
                           padding='VALID', stride=[1, 1],
                           bn=False, is_training=is_training,
                           scope=scope + 'graph_mlp_ml2', bn_decay=bn_decay,
                           activation_fn=tf.identity, is_dist=True)
        #X = X[:, :, 0, :]

        mm, vv = tf.nn.moments(X, [-2], keep_dims=True)
        X = tf.nn.batch_normalization(X, mm, vv, 0, 1, 1e-6)

        adj =tf.expand_dims(tf_util.pairwise_distance(X), axis=0)

        theta = tf_util.conv2d(adj, 250, [1, 1],
                           padding='VALID', stride=[1, 1],
                           bn=False, is_training=is_training,
                           scope=scope + 'graph_theta_ml2', bn_decay=bn_decay,
                           activation_fn=tf.identity, is_dist=True)


        theta = tf_util.conv2d(theta, 50, [1, 1],
                           padding='VALID', stride=[1, 1],
                           bn=False, is_training=is_training,
                           scope=scope + 'graph_theta_ml3', bn_decay=bn_decay,
                           activation_fn=tf.identity, is_dist=True)

        theta = tf_util.conv2d(theta, 1, [1, 1],
                           padding='VALID', stride=[1, 1],
                           bn=False, is_training=is_training,
                           scope=scope + 'graph_theta_ml4', bn_decay=bn_decay,
                           activation_fn=tf.identity, is_dist=True)
        adj = tf.nn.relu(adj-theta)

        X = tf.matmul(tf.squeeze(adj,axis=0),tf.squeeze(X, axis=2))
        #Grad_check = tf.gradients(X,adj, grad_ys=None, name='gradients', gate_gradients=False,
        #                          aggregation_method=None, stop_gradients=None,
        #                          unconnected_gradients=tf.UnconnectedGradients.NONE)
    net = tf.expand_dims(X,axis=2)

    for ci in range(len(outchs)):
        net = tf_util.conv2d(net, outchs[ci], [1, 1],
                             padding='VALID', stride=[1, 1],
                             bn=True, is_training=is_training, weight_decay=weight_decay,
                             scope=scope + '_adj_conv%d' % ci, bn_decay=bn_decay, is_dist=True)

    print('----------------LAYER---------------')
    print(data)

    Grad_check = 0
    return net, adj, adj, theta


def get_model(point_cloud, knn_data=None, is_training=True, bn_decay=None, num_classes=3, param=None):
    # network parameters
    edgconv_layers = [[16], [8], [-8]]
    fc_layers = [64, 32]
    knnlayers = [4, 4, 4]
    k = 5
    pooling = tf.reduce_sum
    if not param is None:
        edgconv_layers = param.edgconv_layers
        fc_layers = param.fc_layers
        knnlayers = param.knnlayers
        k = param.k
        pooling = {'sum': tf.reduce_sum, 'max': tf.reduce_max}[param.pooling]

    """ ConvNet baseline, input is BxNx9 gray image """
    batch_size = point_cloud.get_shape()[0].value
    num_point = point_cloud.get_shape()[1].value
    input_image = tf.expand_dims(point_cloud, -1)
    weight_decay = 0.0

    P = None
    net = tf.reshape(input_image, [1, 564,1,30])
    nets = []
    ps = []
    P1s = []
    for i, outsizes in enumerate(edgconv_layers):
        if outsizes[
            0] > 0:  # interpret values lt 0 as layers where to skip the knn step (an use the one of the pevious step)
            P = None
        outsizes = [abs(v) for v in outsizes]
        net, p, P, theta = dgcnn_layer(net, knnlayers[i], outsizes, k=k, scope=('dgcnn%d' % i), is_training=is_training,
                                bn_decay=bn_decay, P=P)
        P1 = P
        nets.append(net)
        ps.append(p)
        if i == 0:
            P1s = P1

        else:
            P1s = tf.concat([P1s, P1], axis=0)

    out7 = tf_util.conv2d(tf.reshape(tf.concat(nets, axis=-1), [1,564,1,16]), fc_layers[0], [1, 1],
                          padding='VALID', stride=[1, 1],
                          bn=True, is_training=is_training,
                          scope='adj_conv7', bn_decay=bn_decay, is_dist=True)

    out_max = tf_util.max_pool2d(out7, [num_point, 1], padding='VALID', scope='maxpool')

    expand = tf.tile(out_max, [1, num_point, 1, 1])

    #concat = tf.concat(axis=3, values=[expand] + tf.reshape(nets,[1,564,1,32]))

    # CONV
    #net = concat
    for i, fc in enumerate(fc_layers[1:]):
        net = tf_util.conv2d(net, fc, [1, 1], padding='VALID', stride=[1, 1],
                             bn=True, is_training=is_training, scope='dp%d' % i, is_dist=True)
        # net = tf_util.dropout(net, keep_prob=0.5, is_training=is_training, scope='dp1')

    net = tf_util.conv2d(net, num_classes, [1, 1], padding='VALID', stride=[1, 1],
                         activation_fn=None, scope='seg/conv3', is_dist=True)
    net = tf.squeeze(net, [2])

    P = tf.stack(ps)


    return net, P, P1s, theta


def masked_softmax_cross_entropy(preds, labels, mask, node_weights, num_classes):
    """Softmax cross-entropy loss with masking."""
    #     labels = tf.reshape(labels, [871,num_classes])
    print(preds)
    loss = tf.nn.softmax_cross_entropy_with_logits(logits=preds, labels=labels)
    # loss  = tf.reduce_mean(tf.squared_difference(preds, labels))
    mask = tf.cast(mask, dtype=tf.float32)
    mask /= tf.reduce_mean(mask)
    weight_mask = node_weights * mask
    loss *= weight_mask

    return loss


def get_loss(pred, label, mask, node_weights, P):
    num_classes = pred.shape[-1]
    loss = masked_softmax_cross_entropy(pred, label, mask, node_weights, num_classes)

    # per class accuracy
    hard_res = per_instance_seg_pred_res = tf.argmax(pred, 2)
    corr_res = per_instance_seg_pred_res = tf.argmax(label, 2)
    corr_pred = tf.cast(tf.equal(hard_res, corr_res), tf.float32)
    wron_pred = 1 - corr_pred
    P = tf.reduce_mean(tf.log(0.1 + P * 0.9), [0, 3])

    unique_idx, unique_back = tf.unique(tf.reshape(corr_res, (-1,)))
    class_mask = tf.cast(label, tf.float32)
    class_mask = class_mask * mask[None, :, None]
    # print(class_mask)
    per_class_acc = tf.reduce_sum(corr_pred[..., None] * class_mask, [0, -2]) / tf.reduce_sum(class_mask, [0, -2])

    perpoint_weight = tf.gather(per_class_acc, unique_back)
    perpoint_weight = perpoint_weight[None, :]
    # perpoint_weight = tf.reshape(perpoint_weight,(corr_res.shape[0],corr_res.shape[1]))
    samp_loss = (-1 * tf.reduce_sum(corr_pred * P * (1 - perpoint_weight) * mask) + \
                 2 * tf.reduce_sum(wron_pred * P * perpoint_weight * mask)) / tf.reduce_sum(mask)

    #   print(loss)
    #   aaa
    return samp_loss, tf.reduce_sum(loss)
