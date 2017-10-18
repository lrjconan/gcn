import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
from scipy.sparse.linalg.eigen.arpack import eigsh
import os
import sys
import cPickle as pickle
from collections import defaultdict

seed = 1234
# seed = 1243
# seed = 1324
# seed = 1342
# seed = 1423
# seed = 1432
# seed = 2134
# seed = 2143
# seed = 2341
# seed = 2314

diel_split_idx = 0
label_rate = 0.001
use_rand_split = False

def load_sparse_csr(filename):
    loader = np.load(filename)
    return sp.csr_matrix(
      (loader['data'], loader['indices'], loader['indptr']),
      shape=loader['shape'],
      dtype=np.float32)

def read_cites(filename):
    print 'reading cites'
    cites, s_graph = [], defaultdict(list)
    for i, line in enumerate(open(filename)):
        if i % 100000 == 0:
            print 'reading cites {}'.format(i)
        inputs = line.strip().split()
        cites.append((inputs[1], inputs[2]))
        s_graph[inputs[2]].append(inputs[1])
        s_graph[inputs[1]].append(inputs[2])
    return cites, s_graph


def read_sim_dict(filename):
    print 'reading sim_dict'
    sim_dict = defaultdict(list)
    for i, line in enumerate(open(filename)):
        inputs = line.strip().split()
        sim_dict[inputs[0]].append(inputs[1])
    return sim_dict

def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


def gen_split(idx_all, num_train, num_test, num_val=500):
    npr = np.random.RandomState(seed)
    num_total = len(idx_all)

    perm_idx = npr.permutation(num_total)
    idx_train = idx_all[perm_idx[:num_train]]
    idx_val = idx_all[perm_idx[num_train:num_train+num_val]]
    idx_test = idx_all[perm_idx[num_train+num_val:num_train+num_val+num_test]]

    return idx_train, idx_val, idx_test

def load_data(dataset_str):
    """Load data."""

    if dataset_str == "diel":
        data_folder = "data/diel_data/diel"
        
        x = load_sparse_csr(
            os.path.join(data_folder, "{}".format(diel_split_idx), "{}.x.npz".format(diel_split_idx)))
        tx = load_sparse_csr(
            os.path.join(data_folder, "{}".format(diel_split_idx), "{}.tx.npz".format(diel_split_idx)))
        y = np.load(
            os.path.join(data_folder, "{}".format(diel_split_idx), "{}.y.npy".format(diel_split_idx)))
        ty = np.load(
            os.path.join(data_folder, "{}".format(diel_split_idx), "{}.ty.npy".format(diel_split_idx)))

        features = sp.vstack([x, tx], format="csr").tolil()
        labels = np.vstack((y, ty))
        
        graph, id2index = pickle.load(open(os.path.join(data_folder, "{}".format(diel_split_idx), "{}_graph.p".format(diel_split_idx))))
        train_list = pickle.load(open(os.path.join(data_folder, "{}".format(diel_split_idx), "{}_train_list.p".format(diel_split_idx))))
        test_list = pickle.load(open(os.path.join(data_folder, "{}".format(diel_split_idx), "{}_test_list.p".format(diel_split_idx))))
        test_cov = pickle.load(open(os.path.join(data_folder, "{}".format(diel_split_idx), "{}_test_cov.p".format(diel_split_idx))))
        cites, s_graph = read_cites(data_folder + '/hasItem.cfacts')
        sim_dict = read_sim_dict(data_folder + '/sim.dict')

        idx_train = [id2index[xx] for xx in train_list]  
        idx_val = idx_train[:500]
        test_idx_reorder = [id2index[xx] for xx in test_list]
        test_idx_range = np.sort(test_idx_reorder)
        idx_all = np.array(idx_train + test_idx_reorder)

        adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

        idx_nodes = sorted(graph.keys())
        num_nodes = len(idx_nodes)
        dim_feat = features.shape[1]                
        features_all = sp.rand(num_nodes, dim_feat, density=0.0, format='lil', dtype=np.float32)
        features_all[idx_all, :] = features        
        labels_all = np.zeros([num_nodes, labels.shape[1]], dtype=np.int32)
        labels_all[idx_all, :] = labels
        features = features_all
        labels = labels_all

        train_mask = sample_mask(idx_train, labels.shape[0])
        val_mask = sample_mask(idx_val, labels.shape[0])
        test_mask = sample_mask(idx_test, labels.shape[0])

        y_train = np.zeros(labels.shape)
        y_val = np.zeros(labels.shape)
        y_test = np.zeros(labels.shape)
        y_train[train_mask, :] = labels[train_mask, :]
        y_val[val_mask, :] = labels[val_mask, :]
        y_test[test_mask, :] = labels[test_mask, :]        
    else:
        if dataset_str == 'nell':
            dataset_str += '.{}'.format(label_rate)

        names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
        objects = []
        for i in range(len(names)):
            with open("data/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
                if sys.version_info > (3, 0):
                    objects.append(pkl.load(f, encoding='latin1'))
                else:
                    objects.append(pkl.load(f))

        x, y, tx, ty, allx, ally, graph = tuple(objects)
        test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset_str))
        test_idx_range = np.sort(test_idx_reorder)

        if dataset_str == 'citeseer':
            # Fix citeseer dataset (there are some isolated nodes in the graph)
            # Find isolated nodes, add them as zero-vecs into the right position
            test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
            tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
            tx_extended[test_idx_range-min(test_idx_range), :] = tx
            tx = tx_extended
            ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
            ty_extended[test_idx_range-min(test_idx_range), :] = ty
            ty = ty_extended

        features = sp.vstack((allx, tx)).tolil()
        labels = np.vstack((ally, ty))
        adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

        if dataset_str == 'nell.{}'.format(label_rate):
            idx_nodes = sorted(graph.keys())
            num_nodes = len(idx_nodes)
            idx_entity = np.array(range(len(ally)) + test_idx_reorder)
            idx_relation = np.array(list(set(idx_nodes) - set(idx_entity.tolist())))
            one_hot_dim = len(idx_relation)
            dim_feat_raw = features.shape[1]
            dim_feat = one_hot_dim + dim_feat_raw
            
            features_all = np.zeros([num_nodes, dim_feat], dtype=np.float32)
            features_all[idx_entity, :dim_feat_raw] = features.toarray()
            features_all[idx_relation, dim_feat_raw:] = np.eye(one_hot_dim)
            features_all = sp.lil_matrix(features_all)

            labels_all = np.zeros([num_nodes, labels.shape[1]], dtype=np.int32)
            labels_all[idx_entity, :] = labels

            features = features_all
            labels = labels_all
        else:
            features[test_idx_reorder, :] = features[test_idx_range, :]
            labels[test_idx_reorder, :] = labels[test_idx_range, :]

        if use_rand_split:
            # random split
            num_train = len(y)
            num_test = len(test_idx_reorder)
            idx_all = np.array(range(len(ally)) + test_idx_reorder)
            idx_train, idx_val, idx_test = gen_split(idx_all, num_train, num_test, num_val=500)
        else:
            # fix split
            idx_test = test_idx_range.tolist()
            idx_train = range(len(y))
            idx_val = range(len(y), len(y)+500)        

        train_mask = sample_mask(idx_train, labels.shape[0])
        val_mask = sample_mask(idx_val, labels.shape[0])
        test_mask = sample_mask(idx_test, labels.shape[0])

        y_train = np.zeros(labels.shape)
        y_val = np.zeros(labels.shape)
        y_test = np.zeros(labels.shape)
        y_train[train_mask, :] = labels[train_mask, :]
        y_val[val_mask, :] = labels[val_mask, :]
        y_test[test_mask, :] = labels[test_mask, :]

    return adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask


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


def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return sparse_to_tuple(features)


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    return sparse_to_tuple(adj_normalized)


def construct_feed_dict(features, support, labels, labels_mask, placeholders):
    """Construct feed dictionary."""
    feed_dict = dict()
    feed_dict.update({placeholders['labels']: labels})
    feed_dict.update({placeholders['labels_mask']: labels_mask})
    feed_dict.update({placeholders['features']: features})
    feed_dict.update({placeholders['support'][i]: support[i] for i in range(len(support))})
    feed_dict.update({placeholders['num_features_nonzero']: features[1].shape})
    return feed_dict


def chebyshev_polynomials(adj, k):
    """Calculate Chebyshev polynomials up to order k. Return a list of sparse matrices (tuple representation)."""
    print("Calculating Chebyshev polynomials up to order {}...".format(k))

    adj_normalized = normalize_adj(adj)
    laplacian = sp.eye(adj.shape[0]) - adj_normalized
    largest_eigval, _ = eigsh(laplacian, 1, which='LM')
    scaled_laplacian = (2. / largest_eigval[0]) * laplacian - sp.eye(adj.shape[0])

    t_k = list()
    t_k.append(sp.eye(adj.shape[0]))
    t_k.append(scaled_laplacian)

    def chebyshev_recurrence(t_k_minus_one, t_k_minus_two, scaled_lap):
        s_lap = sp.csr_matrix(scaled_lap, copy=True)
        return 2 * s_lap.dot(t_k_minus_one) - t_k_minus_two

    for i in range(2, k+1):
        t_k.append(chebyshev_recurrence(t_k[-1], t_k[-2], scaled_laplacian))

    return sparse_to_tuple(t_k)
