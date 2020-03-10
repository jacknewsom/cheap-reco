import numpy as np
import scipy as sp
from sklearn.cluster import DBSCAN

best_epsilon = 8
best_min_samples = 8

def dbscan(data, epsilon, min_samples):
    '''Simple wrapper around sklearn.cluster.DBSCAN
    '''
    return DBSCAN(eps=epsilon, min_samples=min_samples, metric='euclidean').fit(data).labels_

def find_signal_and_noise(cluster_labels):
    ''' points labeled noise (i.e. points with labels -1)
    '''
    return np.where(cluster_labels != -1), np.where(cluster_labels == -1)

def find_big_and_small_clusters(cluster_labels, cut_size):
    '''Returns indices of voxels in clusters that are larger
    than cut_size.

    Keyword arguments:
    cluster_labels -- (N,) numpy array of voxelwise cluster labels
    cut_size -- minimum number of members a cluster needs to not
                be cut
    '''
    unique, sizes = np.unique(cluster_labels, return_counts=True)
    big_clusters = np.isin(cluster_labels, unique[sizes > cut_size])
    small_clusters = np.isin(cluster_labels, unique[sizes <= cut_size])
    return big_clusters, small_clusters

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

    # separate signal and noise points
    signal_clusters, noise_clusters = find_signal_and_noise(cluster_predictions)
    denoised_data, denoised_features = data[signal_clusters], features[signal_clusters]
    denoised_predictions, denoised_labels = cluster_predictions[signal_clusters], labels[signal_clusters]
    
    noise_data, noise_features = data[noise_clusters], features[noise_clusters]
    noise_predictions, noise_labels = cluster_predictions[noise_clusters], labels[noise_clusters]
    
    # separate large and small clusters
    big_clusters, small_clusters = find_big_and_small_clusters(denoised_predictions, cut_size)
    big_data, big_features = denoised_data[big_clusters], denoised_features[big_clusters]
    big_predictions, big_labels = denoised_predictions[big_clusters], denoised_labels[big_clusters]

    small_data, small_features = denoised_data[small_clusters], denoised_features[small_clusters]
    small_predictions, small_labels = denoised_predictions[small_clusters], denoised_labels[small_clusters]

    all_data = [big_data, small_data, noise_data]
    all_labels = [big_labels.reshape(-1), small_labels.reshape(-1), noise_labels.reshape(-1)]
    all_features = [big_features.reshape(-1), small_features.reshape(-1), noise_features.reshape(-1)]
    all_predictions = [big_predictions, small_predictions, noise_predictions]
    
    return all_data, all_labels, all_features, all_predictions
