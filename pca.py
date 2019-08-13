from sklearn.decomposition import PCA
import numpy as np
import math

def t(p, q, r):
    ''' Helper function for vertex association
    Purpose: Calculate t param (describes closest point on a line to vertex r)
    p : cord1 of pc
    q : cord2 of pc
    r : vertex '''
    x = p-q
    return np.dot(r-q, x)/np.dot(x, x)

def dist(p, q, r):
    ''' Helper function for vertex association
    Purpose: Calculate distance from vertex r to the line traced from p to q'''
    return np.linalg.norm(t(p, q, r)*(p-q)+q-r)

def associate_vertex(p, q, V):
    ''' Finds the closest vertex to given pc
    p -- cord1 of pc
    q -- cord2 of pc
    V -- List of all vertices '''
    distances = [dist(p, q, r) for r in V]
    min_dist = min(distances)
    vertex_idx = distances.index(min_dist)    
    return V[vertex_idx]

def pca(H, V):
    ''' Principle Component Analysis
    Purpose: Find Principle Component of a cluster of hit points

    Input
    H : hit points matrix
    V : List of all vertices 

    Output
    Cordinates of associated vertex'''
    
    pca = PCA(n_components=3)
    pca.fit(H)
    pc_x = pca.components_[:, 0]
    pc_y = pca.components_[:, 1]
    pc_z = pca.components_[:, 2]
    pc_mag = pca.explained_variance_[0]
                             
    p = np.array([pca.mean_[0], pca.mean_[1], pca.mean_[2]])
    q = np.array([pca.mean_[0] + pc_x[0] * math.sqrt(pc_mag),
                  pca.mean_[1] + pc_y[0] * math.sqrt(pc_mag),
                  pca.mean_[2] + pc_z[0] * math.sqrt(pc_mag)])

    # Calculate which vertex is closest
    return associate_vertex(p, q, V)
