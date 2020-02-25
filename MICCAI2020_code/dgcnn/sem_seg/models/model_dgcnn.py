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
  #labels_pl = tf.placeholder(tf.int32, shape=(batch_size, num_point, num_classes))
  return pointclouds_pl, labels_pl

# dgcnn_layer(data, knnchs, outchs,          k=20, scope='', nn_idx=None, is_training=True, bn_decay=0.9,knn_data=None)
def dgcnn_layer(data, knnchs, outchs=[64, 64], k=20, scope='', is_training=True, bn_decay=0.9, weight_decay=0, nn_idx=None, knn_data=None, P=None):
    if knn_data is None:
        knn_data=data
      
    adj=P
    if adj is None:
        adj = tf_util.pairwise_distance(knn_data)
    adj = tf.identity(adj,name=scope+'P')
    
    nn_idx = tf_util.knn(adj, k=k) # (batch, num_points, k)
    edge_feature = tf_util.get_edge_feature(data, nn_idx=nn_idx, k=k)
    Grad_check = tf.gradients(edge_feature,adj , grad_ys=None, name='gradients', gate_gradients=False,
                              aggregation_method=None, stop_gradients=None,unconnected_gradients=tf.UnconnectedGradients.NONE)
    net = edge_feature
    for ci in range(len(outchs)):
        net = tf_util.conv2d(net, outchs[ci], [1,1],
                           padding='VALID', stride=[1,1],
                           bn=True, is_training=is_training, weight_decay=weight_decay,
                           scope=scope+'_adj_conv%d'%ci, bn_decay=bn_decay, is_dist=True)
    
    net_1 = tf.reduce_max(net, axis=-2, keep_dims=True)
    
    print('----------------LAYER---------------')
    print(data)
    print(net_1)
    
    
    return net_1, tf.ones((1,1,1,1),dtype='float32'), nn_idx, Grad_check


def get_model(point_cloud,  knn_data=None, is_training=True, bn_decay=None, num_classes=3, param=None):  
  #network parameters
  edgconv_layers=[[16],[8],[-8]]
  fc_layers=[64, 32]
  knnlayers=[4, 4, 4]
  k=5
  pooling = tf.reduce_sum
  if not param is None:
    edgconv_layers = param.edgconv_layers
    fc_layers = param.fc_layers
    knnlayers = param.knnlayers
    k = param.k
    pooling = {'sum':tf.reduce_sum,'max':tf.reduce_max}[param.pooling]
    
    
  """ ConvNet baseline, input is BxNx9 gray image """
  batch_size = point_cloud.get_shape()[0].value
  num_point = point_cloud.get_shape()[1].value
  input_image = tf.expand_dims(point_cloud, -1)
  weight_decay=0.0
  
  P = None
  net = input_image
  nets = []
  ps = []
  P1s = []
  for i,outsizes in enumerate(edgconv_layers):
    if outsizes[0]>0:#interpret values lt 0 as layers where to skip the knn step (an use the one of the pevious step)
      P = None
    outsizes = [abs(v) for v in outsizes]
    net, p, P, Grad_check  = dgcnn_layer(net, knnlayers[i], outsizes, k=k, scope=('dgcnn%d' % i), is_training=is_training, bn_decay=bn_decay, P=P)
    P1 = P
    nets.append(net)
    ps.append(p)
    if i==0:
        P1s = P1

    else:
        P1s = tf.concat([P1s, P1],axis=0)

   

  out7 = tf_util.conv2d(tf.concat(nets, axis=-1), fc_layers[0], [1, 1], 
                       padding='VALID', stride=[1,1],
                       bn=True, is_training=is_training,
                       scope='adj_conv7', bn_decay=bn_decay, is_dist=True)

  out_max = tf_util.max_pool2d(out7, [num_point, 1], padding='VALID', scope='maxpool')

  expand = tf.tile(out_max, [1, num_point, 1, 1])

  concat = tf.concat(axis=3, values=[expand] + nets)

  # CONV 
  net=concat
  for i,fc in enumerate(fc_layers[1:]):
      net = tf_util.conv2d(net, fc, [1,1], padding='VALID', stride=[1,1],
                 bn=True, is_training=is_training, scope='dp%d' % i, is_dist=True)
      #net = tf_util.dropout(net, keep_prob=0.5, is_training=is_training, scope='dp1')

  net = tf_util.conv2d(net, num_classes, [1,1], padding='VALID', stride=[1,1],
             activation_fn=None, scope='seg/conv3', is_dist=True)
  net = tf.squeeze(net, [2])

  P = tf.stack(ps)

  # total_parameters = 0
  # for variable in tf.trainable_variables():
  #     # shape is an array of tf.Dimension
  #     shape = variable.get_shape()
  #     # print(shape)
  #     # print(len(shape))
  #     variable_parameters = 1
  #     for dim in shape:
  #         variable_parameters *= dim.value
  #     #print(variable_parameters)
  #     total_parameters += variable_parameters
  # print('total_parameters')
  # print(total_parameters)

  return net, P, P1s, Grad_check




def masked_softmax_cross_entropy(preds, labels, mask, node_weights,num_classes):
    """Softmax cross-entropy loss with masking."""
#     labels = tf.reshape(labels, [871,num_classes])
    print(preds)
    loss = tf.nn.softmax_cross_entropy_with_logits(logits=preds, labels=labels)
    #loss  = tf.reduce_mean(tf.squared_difference(preds, labels))
    mask = tf.cast(mask, dtype=tf.float32)
    mask /= tf.reduce_mean(mask)
    weight_mask = node_weights * mask
    loss *= weight_mask

    return loss


def get_loss1(pred, label, mask, node_weights, P):
    """ pred: B,N,13; label: B,N """
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
    # return samp_loss, tf.reduce_sum(loss)
    return samp_loss, tf.reduce_sum(loss)



def get_loss(pred, label, mask, node_weights, P):
  """ pred: B,N,13; label: B,N """
  num_classes = pred.shape[-1]
  loss = masked_softmax_cross_entropy(pred, label,mask,node_weights,num_classes)
  return tf.identity(float(0)), tf.reduce_mean(loss*tf.reduce_mean(tf.log(P+1),[0,3]))