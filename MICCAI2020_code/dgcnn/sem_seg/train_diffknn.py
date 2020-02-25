## This code is written for 5 baselines 0: imaging features , 1: non-imaging features, 2:
import argparse
import math
import h5py
import numpy as np
import tensorflow as tf
import networkx as nx
import matplotlib.pyplot as plt
# tf.get_logger().setLevel('INFO')

import socket
from sklearn.preprocessing import label_binarize
import pandas as pd
import scipy.sparse as sp
import os
import sys
import importlib
import pickle
import datetime
import ast
from mpl_toolkits.mplot3d import Axes3D
from sklearn.linear_model import RidgeClassifier
import scipy

BASE_DIR = '/home/lcosmo/PROJECTS/dgcnn/sem_seg'
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))

from utils import *
import sem_seg.ABIDEParser_awa as Reader
import provider
import provider_tadpole
import utils.tf_util

from visualize import *

from sklearn.model_selection import StratifiedKFold
from numpy import zeros, newaxis
import sklearn.metrics

from utils import *

splits = 10

skf = StratifiedKFold(n_splits=splits, shuffle=False)

parser = argparse.ArgumentParser()
parser.add_argument('--gpu_id', type=int, default=0, help='Gpu id')

# Optimization parameters
parser.add_argument('--max_epoch', type=int, default=100, help='Epoch to run [default: 50]')

# Dataset and features parameters
parser.add_argument('--Dataset', type=int, default=3, help='Tadpole or ABIDE')
parser.add_argument('--features_types', default=0,
                    help='choose out of multi-modal features default=0 for imaging features, 1 for non_imaging, 2: concat')
parser.add_argument('--knn_features_types', default=0,
                    help='choose out of multi-modal features 0 for imaging features, default=1 for non_imaging, 2: concat')

# Network config parameters:
parser.add_argument('--model', default='model_dgcnn_mlp', help='Batch Normalization for convolution layers')
parser.add_argument('--edgconv_layers', type=ast.literal_eval, default=[[16]],help='list of list of fc layers. Negative values means that the knn graph is not computed in that layer')
parser.add_argument('--fc_layers', type=ast.literal_eval, default=[32, 16],help='list of fully connected layers size for classification')
parser.add_argument('--knnlayers', type=ast.literal_eval, default=[8, 8, 8], help='list of knn layers size')


parser.add_argument('--k', type=int, default=5, help='k nearest neighbors')
parser.add_argument('--pooling', default='sum', help='sum or max pooling of the conv layer')
parser.add_argument('--embedding_dim', type=int, default=30,
                    help='dimesion of the feature dimesnionality reduction space')
# parser.add_argument('--batchNorm', default=True, help='Batch Normalization for convolution layers')

FLAGS = parser.parse_args()

max_num_features = FLAGS.embedding_dim

MODEL = importlib.import_module('models.' + FLAGS.model)  # import network module
LOG_DIR = 'log/'
BATCH_SIZE = 1
MAX_EPOCH = FLAGS.max_epoch
# FEATURE_TYPE = FLAGS.features_types
DATASET = FLAGS.Dataset

if not os.path.exists(LOG_DIR): os.mkdir(LOG_DIR)
os.system('cp model.py %s' % (LOG_DIR))
os.system('cp train.py %s' % (LOG_DIR))
LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'w')
LOG_FOUT.write(str(FLAGS) + '\n')


def log_string(out_str):
    LOG_FOUT.write(out_str + '\n')
    LOG_FOUT.flush()
    # print(out_str)


def get_learning_rate(step):
    boundaries2 = [100, 180, 200]
    values2 = [0.001, 0.0001, 0.0005, 0.0001]
    rateMLP = tf.train.piecewise_constant(step, boundaries2, values2)

    return rateMLP


BN_INIT_DECAY = 0.5
BN_DECAY_DECAY_RATE = 0.5
BN_DECAY_DECAY_STEP = 1e4
BN_DECAY_CLIP = 0.99


def get_bn_decay(batch):
    bn_momentum = tf.train.exponential_decay(
        BN_INIT_DECAY,
        batch * BATCH_SIZE,
        BN_DECAY_DECAY_STEP,
        BN_DECAY_DECAY_RATE,
        staircase=True)
    bn_decay = tf.minimum(BN_DECAY_CLIP, 1 - bn_momentum)
    return bn_decay


def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


def get_train_test_masks(labels, y_data, idx_train, idx_val, idx_test, num_classes):
    train_mask = sample_mask(idx_train, labels.shape[0])
    val_mask = sample_mask(idx_val, labels.shape[0])
    test_mask = sample_mask(idx_test, labels.shape[0])

    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_train[train_mask] = labels[train_mask]
    y_val[val_mask] = labels[val_mask]
    y_test[test_mask] = labels[test_mask]
    num_nodes = np.size(labels, 0)
    # print(labels)
    print(num_nodes)
    print(num_classes)
    c = [[i for i in range(num_nodes) if y_data[i][j] == 1] for j in range(num_classes)]

    node_weights = np.zeros(num_nodes)
    for j in range(num_classes):
        node_weights[c[j]] = 1 - (len(c[j]) / float(num_nodes))

    return y_train, y_val, y_test, train_mask, val_mask, test_mask, node_weights


def average_gradients(tower_grads):
    """Calculate average gradient for each shared variable across all towers.

    Note that this function provides a synchronization point across all towers.

    Args:
      tower_grads: List of lists of (gradient, variable) tuples. The outer list
      is over individual gradients. The inner list is over the gradient
      calculation for each tower.
    Returns:
       List of pairs of (gradient, variable) where the gradient has been
       averaged across all towers.
    """
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g, _ in grad_and_vars:
            expanded_g = tf.expand_dims(g, 0)
            grads.append(expanded_g)

        # Average over the 'tower' dimension.
        grad = tf.concat(grads, 0)
        grad = tf.reduce_mean(grad, 0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads


def euclidean_distance(f1, f2):
    diff = f1 - f2
    return np.sqrt(np.dot(diff, diff))


def network_plot_3D(G, angle, save=False):
    # Get node positions
    pos = nx.get_node_attributes(G, 'pos')
    path = G.path
    # Get number of nodes
    n = G.number_of_nodes()

    # Get the maximum number of edges adjacent to a single node
    edge_max = max([G.degree(i) for i in range(n)])

    # Define color range proportional to number of edges adjacent to a single node
    # colors = [plt.cm.plasma(G.degree(i) / edge_max) for i in range(n)]
    colors = G.node_color

    # 3D network plot
    with plt.style.context(('ggplot')):

        fig = plt.figure(figsize=(10, 7))
        ax = Axes3D(fig)

        # Loop on the pos dictionary to extract the x,y,z coordinates of each node
        for key, value in pos.items():
            xi = value[0]
            yi = value[1]
            zi = value[2]

            # Scatter plot
            ax.scatter(xi, yi, zi, c=colors[key], s=20 + 20 * G.degree(key), edgecolors='k', alpha=0.7)
            # ax.scatter(xi, yi, zi, c=colors[key], s=20 , edgecolors='k', alpha=0.7)
        # Loop on the list of edges to get the x,y,z, coordinates of the connected nodes
        # Those two points are the extrema of the line to be plotted
        for i, j in enumerate(G.edges()):
            x = np.array((pos[j[0]][0], pos[j[1]][0]))
            y = np.array((pos[j[0]][1], pos[j[1]][1]))
            z = np.array((pos[j[0]][2], pos[j[1]][2]))

            # Plot the connecting lines
            ax.plot(x, y, z, c='black', alpha=0.5)
            # ax.plot(x, y, z, alpha=0.5)
    # Set the initial view
    ax.view_init(30, angle)

    # Hide the axes
    ax.set_axis_off()

    if save is not False:
        plt.savefig(path + '_3d' + '.png')
        plt.show()
        plt.close()
    else:
        plt.show()

    return


def affinity_visualize2(adj_all, dense_features_, all_labels_, test_mask, num_sample, num_classes, LOG_DIR, epoch,
                        fold):
    graph = nx.Graph()
    num_aff = adj_all.shape[0]
    all_labels = np.where(all_labels_ == 1)[1]
    idx_org = np.where(test_mask == True)
    dense_features_ = np.squeeze(dense_features_, axis=0)
    dense_features_ = dense_features_[idx_org, :]

    all_labels = all_labels[idx_org]
    for nf in range(num_aff):
        adj = adj_all[nf, :, :]
        # adj = np.squeeze(adj, axis=0)
        dense_features = np.squeeze(dense_features_, axis=0)
        num_nodes = dense_features.shape[0]
        # num_nodes = 100
        c = []

        for j in range(num_classes):
            c.append([i for i in range(num_nodes) if all_labels[i] == j])
            # c.append([i for i in range(num_nodes) for k in range(num_nodes) if all_labels[i][k] == i ])
            c[-1] = c[-1][:num_sample]
        idx = np.concatenate(c, axis=0)
        dense_features = dense_features[idx, :]
        all_labels = [all_labels[item] for item in idx]
        num_nodes = len(idx)
        graph.add_nodes_from(np.arange(num_nodes))
        cnt = 0
        threshold = [0.50, 0.96, 0.985]
        for i in range(num_nodes):
            graph.node[i]['pos'] = dense_features[i, 0:3]
            # sp = nx.all_pairs_shortest_path(graph)

            for j in range(i + 1, num_nodes):
                if adj[i, j] >= threshold[nf]:
                    cnt += 1

                    graph.add_edge(i, j,
                                   weight=np.exp(-(euclidean_distance(dense_features[i, :], dense_features[j, :]))))
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                p = dict(nx.single_source_shortest_path_length(graph, i, j))
                # single_source_shortest_path_length
                if graph.has_edge(i, j):
                    graph.remove_edge(i, j)
                    # graph.add_edge(i, j, capacity=15, length=length_[j])
                    graph.add_weighted_edges_from([(i, j, max(p.values()))])
                    # print(max(p.values()))
        node_colors = []
        colors = ['r', 'b', 'g', 'y', 'c', 'm', 'k', 'orange', 'olive', 'grey', 'r', 'b', 'g', 'y', 'c', 'm', 'k',
                  'orange', 'olive', 'grey', 'r', 'b', 'g', 'y', 'c', 'm', 'k', 'orange', 'olive', 'grey']

        # np.random.shuffle(idx)
        for i in range(num_nodes):
            # for i in idx:
            node_colors.append(colors[all_labels[i]])

        # nx.draw_networkx(graph, nx.spring_layout(graph, weight='weight', iterations=50, scale=1000), node_size=5, width=0.1,
        #                 node_color=node_colors, with_labels=False)
        # nx.draw_networkx(graph, pos =nx.spring_layout(graph, weight='weight', iterations=0, scale=10, dim = 2),node_size=10, width=0.2, node_color=node_colors, with_labels=True)
        if nf == 3:  # or nf==2:

            graph.node_color = node_colors
            graph.path = LOG_DIR + '/Graph_Fold_' + str(fold) + 'epoch_' + str(epoch) + 'Layer_' + str(nf) + '.png'
            network_plot_3D(graph, 30, save=True)
        else:

            nx.draw_networkx(graph, pos=dense_features[:, 0:2], node_size=10 + 10 * graph.degree(i), width=0.2,
                             node_color=node_colors, with_labels=False)

            # nx.draw_networkx(adj, pos=dense_features_random, node_size=5, width=0.1,node_color=node_colors, with_labels=False)
            plt.axis('off')
            plt.savefig(LOG_DIR + '/Graph_Fold_' + str(fold) + 'epoch_' + str(epoch) + 'Layer_' + str(nf) + '.png')
            plt.close()
            # plt.show()


def masked_accuracy_regression(preds, labels, mask):
    """Accuracy with masking."""
    # tf.summary.histogram('preds', preds)
    correct_prediction = tf.equal(tf.argmax(preds, 2), tf.argmax(labels, 2))
    accuracy_all = tf.cast(correct_prediction, tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)
    # mask /= tf.reduce_mean(mask)
    # mask1= mask * node_weights
    accuracy_all *= mask
    return tf.reduce_sum(accuracy_all) / tf.reduce_sum(mask)


def masked_accuracy(preds, labels, mask):
    """Accuracy with masking."""
    # tf.summary.histogram('preds', preds)
    correct_prediction = tf.equal(tf.argmax(preds, 2), tf.argmax(labels, 2))
    accuracy_all = tf.cast(correct_prediction, tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)
    # mask /= tf.reduce_mean(mask)
    # mask1= mask * node_weights
    accuracy_all *= mask
    return tf.reduce_sum(accuracy_all) / tf.reduce_sum(mask)


def train_one_epoch(sess, ops, train_writer, current_data, current_label, mask, weight, epoch, fold, LOG_DIR):
    """ ops: dict mapping from string to tf ops """
    is_training = True

    feed_dict = {ops['pointclouds']: current_data[:, :, :],
                 ops['labels']: current_label[None, :, :],
                 ops['is_training']: is_training,
                 ops['mask']: mask,
                 ops['weight']: weight}

    summary, step, _, loss, pred, acc, adj, theta = sess.run(
        [ops['step'], ops['step'], ops['train_op'], ops['loss'], ops['pred'], ops['accuracy'], ops['adj'], ops['theta']],
        feed_dict=feed_dict)

    #     train_writer.add_summary(summary, step)
    #     adj = np.array(adj)
    return acc, loss


def eval_one_epoch(sess, ops, test_writer, current_data, current_label, mask, weight, epoch, fold, LOG_DIR):
    """ ops: dict mapping from string to tf ops """
    is_training = False

    feed_dict = {ops['pointclouds']: current_data[:, :, :],
                 ops['labels']: current_label[None, :, :],
                 ops['is_training']: is_training,
                 ops['mask']: mask,
                 ops['weight']: weight}

    summary, step, loss, pred, acc, adj, theta = sess.run(
        [ops['step'], ops['step'], ops['loss'], ops['pred'], ops['accuracy'], ops['adj'], ops['theta']],
        feed_dict=feed_dict)


    adj = np.array(adj)


    log_string('eval mean loss: %f' % (loss))
    log_string('accuracy: %f' % (acc))

    return acc, loss, adj, theta


LOG_DIR = 'log2/save_model/Adj_viz/' + str(FLAGS.Dataset) + 'Class20_Features_' + str(
    max_num_features) + '/model_' + str(FLAGS.model) + '_feature_' + str(FLAGS.features_types) + '_' + str(
    FLAGS.knn_features_types) + '/' + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')


def feature_selection(features, y, labeled_ind, num_img_features):
    features = Reader.feature_selection(features, y, labeled_ind, num_img_features)
    return features


def load_features(dataset, features_types):
    if dataset == 0:
        with open('TadpoleData.pickle','rb') as f:
            features, y_data, y = pickle.load(f)
    elif dataset == 3:
        x = np.load('/home/leslie/DGCNN/dgcnn/sem_seg/UKBB2_numpy_14504.npy')
        print(x)
        row, col = np.shape(x)
        age = 2020 - x[:, 1].astype(int)
        labels = np.zeros([len(age), 1])
        range_age_low = [50, 60, 70, 80]
        range_age_high = [59, 69, 79, 84]

        for i in range(0, len(range_age_low)):
            y_temp = np.where((age >= range_age_low[i]) & (age <= range_age_high[i]))
            labels[y_temp] = i
        y = labels
        features = x[:, 2:]
        print("max and min of ground truth", max(y), min(y))
        _, fcol = np.shape(features)
        for j in range(0, fcol):
            maxi = max(features[:, j])
            mini = min(features[:, j])
            if maxi != mini:
                for i in range(row):
                    features[i, j] = (features[i, j] - mini) / (maxi - mini)
            else:
                for i in range(row):
                    features[i, j] = 1

        # Get class labels and acquisition site for all subjects
        for i in range(y.shape[0]):
            y[i] = y[i]

        y_data = label_binarize(y, classes=np.arange(6))

        return features, y_data, y
    else:
        file = './Awa_dataset/AwA2-features.txt'
        file_label = './Awa_dataset/AwA2-labels.txt'
        file_idx = './Awa_dataset/Awa_idx20.txt'
        file_attribute = './Awa_dataset/predicate-matrix-binary.txt'
        #         file_idx = './Awa_dataset/AwA2-idx.txt'
        idx = np.loadtxt(file_idx, dtype=int)
        labels = np.loadtxt(file_label, dtype=int)
        labels = labels[idx]
        if features_types == 0:
            if os.path.isfile(file + '.pkl'):
                with open(file + '.pkl', 'rb') as h:
                    features = pickle.load(h)
            else:
                features = np.loadtxt(file)
                with open(file + '.pkl', 'wb') as h:
                    pickle.dump(features, h)
            features = features[idx, :]
        else:
            features_ = np.loadtxt(file_attribute, dtype=int)
            features = np.zeros((len(idx), 85), dtype=float)
            for jj in range(len(labels)):
                features[jj, :] = features_[labels[jj], :]

        row, fcol = np.shape(features)
        for j in range(0, fcol):
            maxi = max(features[:, j])
            mini = min(features[:, j])
            if maxi != mini:
                for i in range(row):
                    features[i, j] = (features[i, j] - mini) / (maxi - mini)
            else:
                for i in range(row):
                    features[i, j] = 1
        labels = labels - 1
        numclasses = len(np.unique(labels))
        # labels = np.squeeze(labels,axis=0)
        y_data = label_binarize(labels, classes=np.arange(numclasses))

        y = labels

    print('Selected dataset: %d' % dataset)
    if dataset == 1:
        if int(features_types) == 0:
            features_, all_labels, one_hot_labels, features, Gender, sites = Reader.load_ABIDE_data(features_types)
        elif int(features_types) == 1:
            features_, all_labels, one_hot_labels, features, Gender, sites = Reader.load_ABIDE_data(features_types)
            features = np.concatenate((np.expand_dims(Gender, axis=1) - 1, np.expand_dims(sites, axis=1)), axis=1)
            features = np.concatenate((features[:, :1], tf.keras.utils.to_categorical(features[:, 1])), 1)
        elif int(features_types) == 2:
            features_, all_labels, one_hot_labels, features, Gender, sites = Reader.load_ABIDE_data(features_types)
            features_1 = np.concatenate((np.expand_dims(Gender, axis=1) - 1, np.expand_dims(sites, axis=1)), axis=1)
            features_1 = np.concatenate((features_1[:, :1], tf.keras.utils.to_categorical(features_1[:, 1])), 1)
            features = np.concatenate((features, features_1), axis=1)

        y = all_labels - 1
        y_data = one_hot_labels

    return features, y_data, y


def train():
    # LOADING FEATURES
    img_features, y_data, y = load_features(FLAGS.Dataset, FLAGS.features_types)
    img_features = img_features.astype(float)
    img_features[np.where(np.isnan(img_features))] = 0
    num_img_features = min(max_num_features, img_features.shape[-1])
    num_nodes = img_features.shape[0]
    num_classes = y_data.shape[1]

    knn_features = np.zeros((num_nodes, 0), 'float32')
    num_knn_features = 0
    if (FLAGS.features_types != FLAGS.knn_features_types):
        knn_features, _, _ = load_features(FLAGS.Dataset, FLAGS.knn_features_types)
        knn_features = knn_features.astype(float)
        knn_features[np.where(np.isnan(knn_features))] = 0
        num_knn_features = min(max_num_features, knn_features.shape[-1])

    img_features_ori = img_features;
    knn_features_ori = knn_features;

    tf.reset_default_graph()
    with tf.Graph().as_default(), tf.device('/gpu:%d' % FLAGS.gpu_id):

        # PLACEHOLDERS
        pointclouds_pl, labels_pl = MODEL.placeholder_inputs(BATCH_SIZE, num_nodes, num_img_features + num_knn_features,
                                                             num_classes)
        # custom placeholders
        is_training_pl = tf.placeholder(tf.bool, shape=())
        mask_pl = tf.placeholder(tf.float32, shape=(num_nodes,))
        node_weights_pl = tf.placeholder(tf.float32, shape=(num_nodes,))

        batch = tf.Variable(0, trainable=False)
        bn_decay = get_bn_decay(batch)
        learning_rate = get_learning_rate(batch)
        trainer = tf.train.AdamOptimizer(learning_rate)

        nodes_features = pointclouds_pl

        # FEATURE DIM RED with MLP layer
        int_loss = tf.identity(float(0))

        # DGCNN MODEL
        pred, P, Padj, theta = MODEL.get_model(nodes_features[:, :, :num_img_features], is_training=is_training_pl,
                                        bn_decay=bn_decay, num_classes=num_classes,
                                        knn_data=nodes_features[:, :, num_img_features:], param=FLAGS)

        #print(Grad_check)
        prom_matrices = [tf.get_default_graph().get_tensor_by_name(n.name + ':0') for n in
                         tf.get_default_graph().as_graph_def().node if n.name in ['dgcnn%dP' % i for i in range(10)]]


        knnloss, loss = MODEL.get_loss(pred, labels_pl, mask_pl, node_weights_pl, P)
        loss = knnloss * 0.01 + int_loss + loss
        accuracy = masked_accuracy(pred, labels_pl, mask_pl)

        #
        grads = trainer.compute_gradients(loss)
        train_op = trainer.apply_gradients(grads, global_step=batch)

        saver = tf.train.Saver(tf.global_variables(), sharded=True, max_to_keep=10)

        # Create a session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        sess = tf.Session(config=config)

        # Add summary writers
        merged = tf.summary.merge_all()

        ops = {'pointclouds': pointclouds_pl,
               'labels': labels_pl,
               'is_training': is_training_pl,
               'pred': pred,
               'loss': loss,
               'accuracy': accuracy,
               'train_op': train_op,
               'merged': merged,
               'weight': node_weights_pl,
               'mask': mask_pl,
               'step': batch,
               'adj': P,
               'Padj': Padj,
               'theta': theta}

        # recycle bin
        prediction = pd.DataFrame({'acc': [0], 'loss': [0]})
        prediction_test = pd.DataFrame({'acc': [0], 'loss': [0]})
        fold = 0
        for train_idxs, test_idxs in list(skf.split(np.zeros(num_nodes), np.squeeze(y))):
            fold = fold + 1

            _, _, _, train_mask, valid_mask, test_mask, node_weights = get_train_test_masks(y,
                                                                                            y_data,
                                                                                            train_idxs,
                                                                                            test_idxs,
                                                                                            test_idxs, num_classes)

            img_features = feature_selection(img_features_ori, y, train_idxs, num_img_features)
            knn_features = knn_features_ori
            if num_knn_features > 0:
                knn_features = feature_selection(knn_features_ori, y, train_idxs, num_knn_features)

            if not os.path.exists(LOG_DIR + '/prediction/acc/'):
                os.makedirs(LOG_DIR + '/prediction/acc/')
                os.makedirs(LOG_DIR + '/prediction/acc_test/')
                os.makedirs(LOG_DIR + '/prediction/loss/')
                os.makedirs(LOG_DIR + '/Curves/')
            if not os.path.exists(LOG_DIR + '/adj/' + str(fold) + '/'):
                os.makedirs(LOG_DIR + '/adj/' + str(fold) + '/')

            train_writer = tf.summary.FileWriter(os.path.join(LOG_DIR + '/Curves/' + '/fold' + str(fold) + '/train/'),
                                                 sess.graph)
            test_writer = tf.summary.FileWriter(os.path.join(LOG_DIR + '/Curves/' + '/fold' + str(fold) + '/test/'),
                                                sess.graph)

            init = tf.group(tf.global_variables_initializer())
            sess.run(init)

            x_data = np.concatenate((img_features, knn_features), 1)
            x_data = np.round(x_data, 16)
            data = x_data
            x_data = x_data[None, :, :]

            clf = RidgeClassifier()
            clf.fit(data[train_idxs, :], y[train_idxs].ravel())
            # Compute the accuracy
            lin_acc = clf.score(data[test_idxs, :], y[test_idxs].ravel())
            for epoch in range(MAX_EPOCH):
                sys.stdout.flush()

                acc, loss = train_one_epoch(sess, ops, train_writer, x_data, y_data, train_mask, node_weights, epoch,
                                            fold, LOG_DIR)
                tf.summary.scalar('acc', acc)
                tf.summary.scalar('loss', loss)
                feed_dict = {ops['pointclouds']: x_data,
                             ops['labels']: np.reshape(y_data, (1, num_nodes, num_classes)),
                             ops['is_training']: False,
                             ops['mask']: train_mask,
                             ops['weight']: node_weights}
                knl, pp = sess.run([knnloss, P], feed_dict=feed_dict)

                feed_dict = {ops['pointclouds']: x_data,
                             ops['labels']: np.reshape(y_data, (1, num_nodes, num_classes)),
                             ops['is_training']: False,
                             ops['mask']: test_mask,
                             ops['weight']: node_weights}
                acte = sess.run(accuracy, feed_dict=feed_dict)
                print('Epoch:%4.2e Accs: %4.2e %4.2e - %.2f %.2f' % (epoch, loss, knl, acc, acte))

                if epoch == MAX_EPOCH - 1:  # or acc>=1-1e-3:
                    # print(pp)
                    log_string('**** EPOCH %03d ****' % (epoch))
                    acc_val, loss_val, adj, theta = eval_one_epoch(sess, ops, test_writer, x_data, y_data, test_mask, node_weights,
                                                       epoch, fold, LOG_DIR)
                    adj = np.squeeze(adj)
                    theta = np.squeeze(theta)
                    np.savetxt(LOG_DIR + '/adj.csv', adj, delimiter=',')
                    np.savetxt(LOG_DIR + '/theta.csv', theta, delimiter=',')
                    tf.summary.scalar('acc_val', acc_val)
                    tf.summary.scalar('loss_val', loss_val)

                    df = pd.DataFrame({'acc': [acc], 'loss': [loss]})
                    prediction.append(df)
                    df3 = pd.DataFrame({'acc': [acc_val], 'loss': [loss_val], 'lin_acc': [lin_acc]})
                    prediction_test.append(df3)

                    writer2 = pd.ExcelWriter(LOG_DIR + '/prediction/acc_test/' + str(fold) + '.xlsx',
                                             engine='xlsxwriter')
                    df3.to_excel(writer2, sheet_name='Sheet1')
                    writer2.save()
                    break


if __name__ == "__main__":
    train()

    #