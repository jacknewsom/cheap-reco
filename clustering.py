import numpy as np
import scipy as sp
from sklearn.cluster import DBSCAN

best_epsilon = 8
best_min_samples = 8

def dbscan(data, epsilon, min_samples):
    '''Simple wrapper around sklearn.cluster.DBSCAN
    '''
    return DBSCAN(eps=epsilon, min_samples=min_samples, metric='euclidean').fit(data).labels_

def find_signal(cluster_labels):
    ''' points labeled noise (i.e. points with labels -1)
    '''
    return np.where(cluster_labels != -1)

def find_big_clusters(cluster_labels, cut_size):
    '''Returns indices of voxels in clusters that are larger
    than cut_size.

    Keyword arguments:
    cluster_labels -- (N,) numpy array of voxelwise cluster labels
    cut_size -- minimum number of members a cluster needs to not
                be cut
    '''
    unique, sizes = np.unique(cluster_labels, return_counts=True)
    big_clusters = np.isin(cluster_labels, unique[sizes > cut_size])
    return big_clusters

def cluster_and_cut(data, labels, features, cut_size):
    '''Clusters data with DBSCAN, then cuts noise
    and clusters with less than `cut_size` members.

    Keyword arguments:
    data -- (N, 3) numpy array of coordinates
    labels -- (N,) numpy array of voxelwise labels
    features -- (N,) numpy array of voxelwise features
    cut_size -- minimum number of members a cluster needs to not
                be cut
    '''
    # Calculate DBSCAN labels
    cluster_predictions = dbscan(data, best_epsilon, best_min_samples)

    # remove noise points
    signal_clusters = find_signal(cluster_predictions)
    denoised_data, denoised_features = data[signal_clusters], features[signal_clusters]
    denoised_predictions, denoised_labels = cluster_predictions[signal_clusters], labels[signal_clusters]

    # remove small clusters
    big_clusters = find_big_clusters(denoised_predictions, cut_size)
    cut_data, cut_features = denoised_data[big_clusters], denoised_features[big_clusters]
    cut_predictions, cut_labels = denoised_predictions[big_clusters], denoised_labels[big_clusters]
    return cut_data, cut_labels.reshape(-1), cut_features.reshape(-1), cut_predictions
