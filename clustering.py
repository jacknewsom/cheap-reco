import numpy as np
from graphutils import dbscan, construct_graph, color_graph, braindead_vertex_association

'''Utilities for clustering and grouping voxelized data.
'''

def group_clusters(data, epsilon=2, min_samples=2):
    '''Takes an NxM array of points. Performs DBSCAN on
    the points to form clusters. Then, connects each cluster
    to its nearest neighbor and colors all connected clusters.
    Returns data in original order plus an extra column for
    color (i.e. an Nx(M+1) array).

    Keyword arguments:
    data -- NxM numpy array
    epsilon -- eps parameter in DBSCAN
    min_samples -- min_samples parameter in DBSCAN
    '''
    labels = dbscan(data, epsilon, min_samples).labels_
    # add original row index to end of row
    row_indices = np.arange(data.shape[0]).reshape((data.shape[0], 1))
    # 3 spatial coordinates + 1 row index
    data = np.hstack((data, row_indices))
    # 3 spatial coordinates + 1 cluster index
    signal_indices = np.where(labels != -1)
    noise_indices = np.where(labels == -1)
    signal = data[signal_indices]
    noise = data[noise_indices]
    clusters = {}
    labels = labels[signal_indices]
    for i, label in enumerate(labels):
        if label not in clusters.keys():
            clusters[label] = []
        clusters[label].append(signal[i])
    for cluster in clusters:
        clusters[cluster] = np.vstack(clusters[cluster])
    clusters = clusters.values()
    graph = construct_graph(clusters)
    color = color_graph(graph)
    colored_signal = {}
    for cluster in graph:
        color = cluster.color.value
        if color not in colored_signal:
            colored_signal[color] =[]
        colored_signal[color].append(cluster.points)
    for color in colored_signal:
        colored_signal[color] = np.vstack(colored_signal[color])
        color_dim = np.full((colored_signal[color].shape[0], 1), color)
        # add cluster index as final column
        colored_signal[color] = np.hstack((colored_signal[color], color_dim))
    colored_signal = np.vstack(colored_signal.values())
    # add cluster index as final column
    colored_noise = np.hstack((noise, np.full((noise.shape[0], 1), -1)))
    useful_indices = np.array([0, 1, 2, 4])
    colored_data = np.empty((colored_signal.shape[0]+colored_noise.shape[0], 4))
    for noisy_point in colored_noise:
        colored_data[int(noisy_point[-2])] = noisy_point[useful_indices]
    for signal_point in colored_signal:
        colored_data[int(signal_point[-2])] = signal_point[useful_indices]
    return colored_data

def simple_cluster_vertex_association(data, vertices):
    '''Returns closest vertex to each pixel. The vertices in
    this function are 'physics' vertices, not the Vertices used
    above. Use this function on the output of group_clusters.

    Keyword arguments:
    data -- (N, 4) numpy array. 3 spatial coordinates + 1 cluster coordinate
    vertices -- list of numpy arrays with 1 row
    '''
    clusters = {}
    colors = {}
    for row in data:
        if row[-1] not in clusters:
            clusters[row[-1]] = []
        clusters[row[-1]].append(row[:-1])
    for cluster in clusters:
        clusters[cluster] = np.vstack(clusters[cluster])
    closest_vertices = braindead_vertex_association(clusters.values(), vertices)
    for cluster, vertex in zip(clusters.keys(), closest_vertices):
        colors[cluster] = vertex
    rowwise_vertices = np.empty((data.shape[0]))
    for i, row in enumerate(data):
        rowwise_vertices[i] = colors[row[-1]]
    return rowwise_vertices

def point_vertex_association(data, vertices, epsilon=2, min_samples=2):
    '''Takes an (N, M) array of points and returns an N long list with
    entries corresponding to closest vertex to each supercluster.

    Acts as a simple wrapper around group_clusters and
    simple_cluster_vertex_association.

    Keyword arguments:
    data -- (N, M) numpy array
    epsilon -- closeness parameter for DBSCAN
    min_samples -- minimum number of samples for DBSCAN to form a cluster
    vertices -- list of numpy arrays with 1 row
    '''
    colored_data = group_clusters(data, epsilon, min_samples)
    return simple_cluster_vertex_association(colored_data, vertices)
