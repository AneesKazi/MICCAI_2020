# In the following code we show DGM model on Tadpole dataset for transductive setting.
# The following code shows the 10 fold cross validation results for the accuracy of classification.
# For this application we have one graph over the entire population and the task is to perform node level classification.

from DGM_model import *
import tensorflow as tf
import numpy as np
# import matplotlib.pyplot as plt
import pickle
from sklearn.linear_model import RidgeClassifier
import sklearn
from sklearn.metrics import f1_score

# from matplotlib import cm
tf.compat.v1.disable_eager_execution()

LOG_DIR = '/home/leslie/DGM-master/results/Tadpole/'


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


tf.compat.v1.reset_default_graph()

batch_size = 1  # Since the whole population is incorporated in one graph.
num_point = 564  # Number of samples in the dataset
num_features = 30  # number of features per sample
num_classes = 3  # Classes : Normal (N), Mild Cognitive Impairment (MCI) and Alzheimer's (AD)

is_training = tf.compat.v1.placeholder_with_default(True, shape=())  # to enable the training phase
pl_mask = tf.compat.v1.placeholder(tf.float32, shape=(
    num_point,))  # To mask out the training and testing samples in transductive setting for during training and testing respectively.

pl_X, pl_y = placeholder_inputs(batch_size, num_point, num_features, num_classes)  # placeholders for training data
pred, adj_hat, theta = get_model(pl_X, num_classes,
                                 is_training)  # Model outputs prediction and sampled probabilistic graph

loss = get_loss(pred, pl_y, pl_mask)  # L_graph as graph loss and node_loss is the Categorical Cross-Entropy

# acc = tf.reduce_sum(tf.cast(tf.equal(tf.argmax(pred, 2), tf.argmax(pl_y, 2)), tf.float32) * pl_mask) / tf.reduce_sum(pl_mask) # accuracy
acc = masked_accuracy(pred, pl_y, pl_mask)
lr = tf.compat.v1.placeholder_with_default(1e-2, shape=None)

opt = tf.compat.v1.train.AdamOptimizer(learning_rate=lr)  # Create an optimizer with the desired parameters.
opt_op1 = opt.minimize(loss)

with open('train_data.pickle', 'rb') as f:
    X_, y_, train_mask_, test_mask_, weight_ = pickle.load(f)  # Load the data
X_ = X_[..., :30]  # For DGM we use modality 1 (M1) for both node representation and graph learning.
print(X_.shape)
y_ = y_[None, ...]  # Labels

acc_10_fold = np.zeros((10, 1))
for fold in range(10):
    # getting the data of the respective fold
    X = np.expand_dims(X_[:, :30, fold], axis=0)
    print(X.shape)
    y = y_[:, :, :, fold]
    print(y.shape)
    train_mask = train_mask_[:, fold]
    test_mask = test_mask_[:, fold]
    weight = np.squeeze(weight_[:, fold])
    # weight = np.zeros((num_point, 1)) + 1
    train_mask = train_mask * weight
    # Intialize the session

    # clf = RidgeClassifier()
    # clf.fit(X[train_mask, :], y[train_mask].ravel())
    # lin_acc = clf.score(X[test_mask, :], y[test_mask].ravel())
    # Compute the AUC
    # pred = clf.decision_function(X[test_mask, :])

    # y_one_hot = label_binarize(y[test_ind], classes=np.arange(3))
    # lin_auc = sklearn.metrics.roc_auc_score(y, pred)
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.compat.v1.Session(config=config)
    sess.run(tf.compat.v1.global_variables_initializer())
    total_parameters = 0
    for variable in tf.compat.v1.trainable_variables():
        # shape is an array of tf.Dimension
        shape = variable.get_shape()
        print(shape)
        print(len(shape))
        variable_parameters = 1
        for dim in shape:
            print(dim)
            variable_parameters *= dim
        # print(variable_parameters)
        total_parameters += variable_parameters
    # print(total_parameters)
    # exit()

    # TRAIN with train_mask
    init_lr = 1e-1  # learning rate
    feed_dict = dict({pl_X: X, pl_y: y, pl_mask: train_mask, is_training: True})
    # Traning for 600 epochs
    for i in range(1000):
        if i % 200 == 0:
            init_lr /= 10
            feed_dict.update({lr: init_lr})

        _, l1, a, adj, pred_ = sess.run([opt_op1, loss, acc, adj_hat, pred], feed_dict=feed_dict)
        if i % 200 == 0:
            print('Iter %d]  Task loss: %.2e,  Acc:%.1f' % (i, l1, a * 100))
            # Testing

            feed_dict_test = dict({pl_X: X, pl_y: y, pl_mask: test_mask, is_training: False})
            l1, a_test, adj, pred_, theta_ = sess.run([loss, acc, adj_hat, pred, theta], feed_dict=feed_dict_test)
            print('TEST       : Task loss: %.2e,  Acc: %.1f' % (l1, a_test * 100))
            # pred_ = np.squeeze(pred_[:,np.argwhere(test_mask != 0), :],axis=0)
            # y_temp =np.squeeze(y[:,np.argwhere(test_mask != 0),:],axis=0)
            # Pred_number = np.argmax(np.squeeze(pred_, axis=1))
            # y_temp_number =  np.argmax(np.squeeze(y_temp, axis=1))
            # auc_ = sklearn.metrics.roc_auc_score(y_temp, np.squeeze(pred_,axis=1))
            # print(np.nonzero(y_temp_number))
            # F1 = f1_score(np.squeeze(y_temp_number,axis=1),  np.squeeze(Pred_number,axis=1), average=None)
            # print(auc_)
            # print(F1)
        if i == 1000:
            adj = sess.run([adj_hat], feed_dict=feed_dict)
            adj = np.squeeze(np.squeeze(np.array(adj), axis=0), axis=0)
            plt.imshow(adj)
            plt.colorbar()
            plt.show()
    acc_10_fold[fold] = a_test
    adj = np.array(adj)
    adj = np.squeeze(np.array(adj), axis=0)
    # plt.imshow(adj,cmap = cm.viridis, vmin=0., vmax=np.max(adj));
    # plt.colorbar()
    # plt.show()
    # plt.close()
    # if not os.path.exists(LOG_DIR + str(fold) + '/'):
    # os.makedirs(LOG_DIR + str(fold) + '/')
    # np.savetxt(LOG_DIR + str(fold) + '/' + 'adj' + str(fold) + '.xlsx', adj, delimiter=',')
    # np.savetxt(LOG_DIR + str(fold) + '/'+'pred' + str(fold) + '.xlsx', np.squeeze(pred_,axis=0), delimiter=',')
    # np.savetxt(LOG_DIR + str(fold) + '/' + 'labels' + str(fold) + '.xlsx', np.argmax(np.squeeze(y, axis=0), axis=1),delimiter=',')
    # np.savetxt(LOG_DIR + str(fold) + '/' + 'theta' + str(fold) + '.xlsx', theta_, delimiter=',')

    # p = 0
    # for i in range(10):
    #    cp = sess.run(pred, feed_dict=feed_dict)
    #   p += cp
    # a = np.sum(np.equal(np.argmax(p, -1), np.argmax(y, -1)) * test_mask) / np.sum(test_mask) # save the accuracies

np.savetxt(LOG_DIR + 'accuracy_withoutweightloss.xlsx', acc_10_fold, delimiter=',')

print('TEST VOTING: Acc: %.1f' % (a * 100))
