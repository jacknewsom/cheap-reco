import numpy as np
import scipy as sp
from sklearn.metrics import pairwise_distances as dist

touching_distance = 1

def is_cluster_touching_vertex(cluster, vertex):
    '''Determines if a cluster is directly touching
    a vertex. Returns a boolean.

    Keyword arguments:
    cluster -- (N, 3) numpy array of coordinates in 3D
    vertex -- (1, 3) numpy array of coordinates in 3D
    '''
    everything = np.vstack((cluster, vertex))
    dist_matr = dist(everything, everything)
    for i in range(len(dist_matr)-1):
        if dist_matr[i, -1] < touching_distance:
            return True
    return False
