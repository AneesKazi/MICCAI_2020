import tf_util
import tensorflow as tf
import numpy as np


def ADGM(features, edges, k, conv_func, is_training, scope=''):
    X_hat = conv_func(features, edges)

    # Probabilistic graph generator
    A = -tf_util.pairwise_distance(X_hat)

    #     if temp is None:
    temp = (1 + tf_util._variable_with_weight_decay(scope + 'temp', [1], 1e-6, None, use_xavier=False))
    #     print('TEMP: ', temp)

    f = features.get_shape().as_list()[-1]
    th = (1 + tf_util._variable_with_weight_decay(scope + 'th', [1, ], 1e-6, None, use_xavier=False))

    A = tf.nn.sigmoid(temp * (A + tf.abs(th)))
    # A = A * (1 - tf.eye(A.get_shape().as_list()[-1]))
    A = A / tf.reduce_sum(A, -1)[..., None]
    return X_hat, A, A


def DGM(features, edges, k, conv_func, is_training, scope=''):
    # The mathematical description of DGM part is detailed in section 2.1 of the manuscript. DGM is divided in three parts as follows

    # part 1: Graph representation feature learning
    X_hat = conv_func(features, edges)

    # Probabilistic graph generator
    D = tf_util.pairwise_distance(X_hat)
    D = D + tf.eye(int(D.shape[-1])) * 1e12
    P = -D  # Converting the distance to probabilities

    # Graph sampling
    edges_hat, probs = sample_without_replacement(P, k, is_training, scope=scope)
    return X_hat, edges_hat, probs


def sample_without_replacement(P, K, is_training, scope=''):
    temp = (1 + tf_util._variable_with_weight_decay(scope + '_temp', [1], 1e-6, None, use_xavier=False))

    P = P * (tf.abs(temp) + 1e-8)
    P = tf.identity(P, name=scope + '_P')

    b, n, f = P.get_shape().as_list()

    #   apply gumble top-k trick
    q = tf.random_uniform(tf.shape(P), 0, 1) + 1e-15  # q=u
    Pq = (P - tf.log(-tf.log(q)))

    _, indices = tf.nn.top_k(Pq, K)

    #   Rearranging the indices
    cols = tf.reshape(tf.tile(tf.tile(tf.range(K)[None, :], (n, 1))[None, :], (b, 1, 1)),
                      [-1])  # fetch the prob for the sampled edges
    rows = tf.tile(tf.tile(tf.range(n)[:, None], (1, K))[None, :], (b, 1, 1))
    rows = tf.reshape((rows + n * np.arange(b)[:, None, None]), [-1])  # .flatten()

    ndindices = tf.concat((rows[:, None], tf.reshape(indices, (-1, 1))), -1)
    samp_logprobs = tf.reshape(tf.gather_nd(tf.reshape(P, (-1, f)), ndindices),
                               (b, n, -1))  # samp_logprobs = log(p(e_ij))

    return indices, tf.exp(samp_logprobs)


def EdgeConv(features, edges, out_chs, k, is_training, dropout=0, weight_decay=0, scope='EdgeConv'):
    features = tf.nn.dropout(features, (1 - dropout * tf.cast(is_training, tf.float32)))

    if k == 0 or edges is None:
        edges = np.arange(features.get_shape().as_list()[1]).astype('int32')[None, :, None]

    net = tf_util.get_edge_feature(features, nn_idx=edges, k=k)

    net = tf_util.conv2d(net, out_chs, [1, 1],
                         padding='VALID', stride=[1, 1],
                         bn=False, is_training=is_training, weight_decay=weight_decay,
                         scope=scope + '_conv', is_dist=True)

    net = tf.reduce_sum(net, axis=-2, keep_dims=True)
    return net


def AEdgeConv(features, edges, out_chs, k, is_training, dropout=0, weight_decay=0, scope='EdgeConv'):
    features = tf.nn.dropout(features, (1 - dropout * tf.cast(is_training, tf.float32)))

    n = features.get_shape().as_list()[1]
    #     edges = np.tile(np.arange(n).astype('int32')[:,None],[1,n]).T[None,...]
    #     print(edges.shape)
    # net = tf_util.get_edge_feature(features, nn_idx=edges, k=n)
    F = tf.tile(features, [1, 1, n, 1])
    net = tf.concat([F, tf.transpose(F, [0, 2, 1, 3])], -1)

    net = tf_util.conv2d(net, out_chs, [1, 1],
                         padding='VALID', stride=[1, 1],
                         bn=False, is_training=is_training, weight_decay=weight_decay,
                         scope=scope + '_conv', is_dist=True)

    #     print(net)
    #     print(edges)
    net = net * edges[..., None]  # /tf.reduce_sum(edges,-2)[:,:,None,None]
    net = tf.reduce_sum(net, axis=-2, keep_dims=True)
    return net


def GraphConv(features, edges, out_chs, k, is_training, dropout=0, weight_decay=0, scope='EdgeConv'):
    #     A = A / tf.reduce_sum(A, -1)[..., None]
    features = tf.nn.dropout(features, (1 - dropout * tf.cast(is_training, tf.float32)))
    if k == 0 or edges is None:
        edges = np.arange(features.get_shape().as_list()[1])[None, :, None]
        net = tf_util.get_node_feature(features, nn_idx=edges, k=1)
    else:
        net = tf_util.get_node_feature(features, nn_idx=edges, k=k) / (k + 1)

    net = tf_util.conv2d(net, out_chs, [1, 1],
                         padding='VALID', stride=[1, 1],
                         bn=False, is_training=is_training, weight_decay=weight_decay,
                         scope=scope + '_conv', is_dist=True)
    net = tf.reduce_sum(net, axis=-2, keep_dims=True)
    return net


def AGraphConv(features, A, out_chs, k, is_training, dropout=0, weight_decay=0, scope='EdgeConv'):
    #     A = A / tf.reduce_sum(A, -1)[..., None]
    if A is None:
        A = tf.eye(features.get_shape().as_list()[1])[None, ...]

    net = tf.nn.dropout(features, (1 - dropout * tf.cast(is_training, tf.float32)))

    net = tf_util.conv2d(net, out_chs, [1, 1], padding='VALID', stride=[1, 1], activation_fn=tf.identity,
                         bn=False, is_training=is_training, scope='dp%d' % np.random.randint(1, 1000), is_dist=True)
    net = tf.nn.relu(tf.matmul(A[0], net[0, :, 0, :]))
    net = tf.expand_dims(tf.expand_dims(net, axis=0), axis=2)
    return net


def Linear(features, A, out_chs, k, is_training, dropout=0, weight_decay=0, scope='EdgeConv'):
    #     A = A / tf.reduce_sum(A, -1)[..., None]
    net = tf.nn.dropout(features, (1 - dropout * tf.cast(is_training, tf.float32)))
    net = tf_util.conv2d(net, out_chs, [1, 1], padding='VALID', stride=[1, 1], activation_fn=tf.identity,
                         bn=False, is_training=is_training, scope='dp%d' % np.random.randint(1, 1000), is_dist=True)
    net = tf.nn.relu(net)
    #     net= tf.expand_dims(tf.expand_dims(net, axis=0),axis=2)
    return net


def MLP(features, A, out_chs, k, is_training, dropout=0, weight_decay=0, scope='EdgeConv'):
    #     A = A / tf.reduce_sum(A, -1)[..., None]
    net = features
    for outch in out_chs[:-1]:
        net = tf.nn.dropout(net, (1 - dropout * tf.cast(is_training, tf.float32)))
        net = tf_util.conv2d(net, outch, [1, 1], padding='VALID', stride=[1, 1], activation_fn=tf.identity,
                             bn=False, is_training=is_training, scope='dp%d' % np.random.randint(1, 1000), is_dist=True)
        net = tf.nn.relu(net)
    net = tf_util.conv2d(net, out_chs[-1], [1, 1], padding='VALID', stride=[1, 1], activation_fn=tf.identity,
                         bn=False, is_training=is_training, scope='dp%d' % np.random.randint(1, 1000), is_dist=True)

    #     net= tf.expand_dims(tf.expand_dims(net, axis=0),axis=2)
    return net
