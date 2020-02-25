import os

import numpy as np
from scipy import sparse as sp
import random
import pandas as pd
from sklearn.linear_model import RidgeClassifier
from sklearn.feature_selection import RFE
import sem_seg.ABIDE_parser as Reader2

# Reading and computing the input data

# Selected pipeline
pipeline = 'cpac'

num_nodes = 564
num_classes = 3
num_features = 30
no_of_aff = 4 #!--
ground_truth = []
gender_=np.zeros((564,1))


root_folder = 'sem_seg/'
#data_folder = os.path.join(root_folder, 'ABIDE_pcp/cpac/filt_noglobal')
#phenotype = os.path.join(root_folder, 'ABIDE_pcp/Phenotypic_V1_0b_preprocessed1.csv')



def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""

    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx


def load_ABIDE_data(features_types):
    Gender = []
    subject_IDs = Reader2.get_ids()
    labels = Reader2.get_subject_score(subject_IDs, score='DX_GROUP')
    Gender_list, Gender = Reader2.get_subject_score(subject_IDs, score='SEX')

    # Get acquisition site
    sites = Reader2.get_subject_score(subject_IDs, score='SITE_ID')
    unique = np.unique(list(sites.values())).tolist()

    num_classes = 2
    num_nodes = len(subject_IDs)

    # Initialise variables for class labels and acquisition sites
    y_data = np.zeros([num_nodes, num_classes])
    y = np.zeros([num_nodes, 1])
    site = np.zeros([num_nodes], dtype=np.int)

    # Get class labels and acquisition site for all subjects
    for i in range(num_nodes):
        y_data[i, int(labels[subject_IDs[i]]) - 1] = 1
        y[i] = int(labels[subject_IDs[i]])
        site[i] = unique.index(sites[subject_IDs[i]])

    # Compute feature vectors (vectorised connectivity networks)
    features = Reader2.get_networks(subject_IDs, kind='correlation', atlas_name='ho')

    #gender_adj = np.zeros((num_nodes, num_nodes))
    #gender_adj = Reader2.create_affinity_graph_from_scores(['SEX'], subject_IDs)
    #site_adj = np.zeros((num_nodes, num_nodes))
    #site_adj = Reader2.create_affinity_graph_from_scores([ 'SITE_ID'], subject_IDs)
    #mixed_adj = gender_adj+ site_adj

    c_1 = [i for i in range(num_nodes) if y[i] == 1]
    c_2 = [i for i in range(num_nodes) if y[i] == 2]

    # print(idx)
    y_data = np.asarray(y_data, dtype=int)
    num_labels = 2
    #one_hot_labels = np.zeros((num_nodes, num_labels))
    #one_hot_labels[np.arange(num_nodes), y_data] = 1
    sparse_features = sparse_to_tuple(sp.coo_matrix(features))
    return sparse_features, y ,y_data,features, Gender, site



def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def feature_selection(matrix, labels, train_ind, fnum):
    """
        matrix       : feature matrix (num_subjects x num_features)
        labels       : ground truth labels (num_subjects x 1)
        train_ind    : indices of the training samples
        fnum         : size of the feature vector after feature selection

    return:
        x_data      : feature matrix of lower dimension (num_subjects x fnum)
    """

    estimator = RidgeClassifier()
    selector = RFE(estimator, fnum, step=100, verbose=1)
    #print(np.shape(train_ind))
    #print(train_ind)
    #print("at ferature selection")

    featureX = matrix[train_ind, :]
    featureY = labels[train_ind]
    selector = selector.fit(featureX, featureY)
    x_data = selector.transform(matrix)


    print("Number of labeled samples %d" % len(train_ind))
    print("Number of features selected %d" % x_data.shape[1])

    return x_data


def get_features(FEATURE_TYPE):
    df = pd.read_excel('Tadpole dataset - Sheet1.xlsx')
    df = np.array(df)
    row,col = np.shape(df)
    ground_truth = df[:,10].astype(int)
    print("max and min of ground truth", max(ground_truth), min(ground_truth))
    for i in range(col):
        if df[1,i] == "1.5 Tesla MRI" or df[1, i] == "3 Tesla MRI":
            for r in range(row):
                if df[r,i]== "1.5 Tesla MRI":
                    df[r,i]= 1.5
                elif df[r,i] == "3 Tesla MRI":
                    df[r, i] = 3
        if df[1, i] == "Pass" or df[1, i] == "Fail":
            for r in range(row):
                if df[r, i] == "Pass":
                    df[r, i] = 1
                elif df[r, i] == "Fail":
                    df[r, i] = 0

    #np.savetxt("groundtruth.csv", ground_truth, delimiter=',')
    features_1 = 0;
    age = np.reshape(df[:, 11], [len(df[:, 11]), 1])
    gender = np.reshape(df[:, 12], [len(df[:, 11]), 1])
    d = {'Male': 1, 'Female': 1}
    for k in range(len(gender)):
        if gender[k] == 'Male':
            gender_[k] = 1
        else:
            gender_[k] = 2
    fdg = np.reshape(df[:, 18], [len(df[:, 11]), 1])
    apoe = np.reshape(df[:, 17], [len(df[:, 11]), 1])
    non_imaging_features = np.concatenate((age, gender_, fdg, apoe), axis=1)
    features = df[:, 19:]

    #features = df[:,18:]
    _,fcol = np.shape(features)
    for j in range(0,  fcol):
        maxi = max(features[:,j])
        mini = min(features[:,j])
        if maxi!=mini:
            for i in range(row):
                features[i, j] = (features[i, j] - mini) / (maxi - mini)
        else:
            for i in range(row):
                features[i, j] = 1
    print(np.shape(features),"shape of features")

    #features = df[:,18:]
    _,fcol = np.shape(non_imaging_features)
    for j in range(0,  fcol):
        maxi = max(non_imaging_features[:,j])
        mini = min(non_imaging_features[:,j])
        if maxi!=mini:
            for i in range(row):
                non_imaging_features[i, j] = (non_imaging_features[i, j] - mini) / (maxi - mini)
        else:
            for i in range(row):
                non_imaging_features[i, j] = 1
    print(np.shape(non_imaging_features),"shape of features")


    if FEATURE_TYPE== 0 or FEATURE_TYPE== 3:
        features_1 = non_imaging_features;

    #Baseline 2, non imaging features
    elif FEATURE_TYPE== 1:
        features = non_imaging_features
    else:
        temp = features
        features = np.concatenate((temp, non_imaging_features), axis=1)


    #exit()

    if FEATURE_TYPE==3:
        return ground_truth, features, features_1
    else:
        return ground_truth, features



def get_affinity(sparse_graph):
    df = pd.read_excel('Tadpole dataset - Sheet1.xlsx')
    df = np.array(df)
    num_nodes, col = np.shape(df)
    age = df[:,11]
    gender = df[:,12]
    fdg = df[:, 18]
    apoe = df[:, 17]
    graph = np.zeros((no_of_aff, num_nodes, num_nodes))
    print("max,min of age: ", max(age), min(age))
    for i in range(num_nodes):
         for j in range(i+1, num_nodes):
             if gender[i] == gender[j]:
                graph[0, i, j] += 1
                graph[0, j, i] += 1
    #np.savetxt("gender.csv", graph[0], delimiter=',')

    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if np.absolute(age[i] - age[j]) < 2:
                graph[1, i, j] += 1
                graph[1, j, i] += 1
    #np.savetxt("age.csv", graph[1], delimiter=',')

    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if np.absolute(fdg[i] - fdg[j]) < 0.10:
                graph[2, i, j] += 1
                graph[2, j, i] += 1
    #np.savetxt("fdg.csv", graph[2], delimiter=',')

    for i in range(num_nodes):
        for j in range(i+1, num_nodes):
            if apoe[i] == apoe[j]:
                graph[3, i, j] += 1
                graph[3, j, i] += 1

    #np.savetxt("apoe.csv", graph[3], delimiter=',')

    for i in range(no_of_aff):
        graph[i] = np.divide(graph[i] - graph[i].min(), graph[i].max() - graph[i].min())

    for i in range(no_of_aff):
        graph[i] = graph[i]*sparse_graph

    return graph

def person_corr_calc(patient_ids, labels):
    #fdg
    #updrs=APOE
    df = pd.read_excel('Tadpole dataset - Sheet1.xlsx')
    df = np.array(df)
    num_nodes, col = np.shape(df)
    age = df[:,11]
    gender = df[:,12]
    gender_ = df[:,12]
    moca= df[:, 18]
    updrs = df[:, 17]
    data = pd.DataFrame()
    moca_vec = []
    updrs_vec = []
    age_vec = []
    gender_vec = []
    print(len(patient_ids))
    print(len(moca_vec))
    for x in range(563):
        #print(moca[x])
        moca_vec.append(moca[x])
        updrs_vec.append(updrs[x])
        age_vec.append(age[x])
        gender_vec.append(gender[x])

    print(len(moca_vec))
    print(len(updrs_vec))
    labels.tolist()
    data['moca'] = moca_vec
    data['updrs'] = updrs_vec
    data['gender'] = gender_vec
    data['age'] = age_vec
    data['label'] = labels


    print(data.corr(method='pearson'))
    print(data.corr(method='spearman'))

