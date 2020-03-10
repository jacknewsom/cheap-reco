from sklearn.decomposition import PCA
import numpy as np
import math

cutoff_distance = 10

def t(p, q, r):
    ''' Helper function for closest_vertex
    Purpose: Calculate t param (describes closest point on a line to vertex r)

    --- Parameters ---
    p - cord1 of pc [1x3]
    q - cord2 of pc [1x3]
    r - vertex [1x3] '''
    x = p-q
    return np.dot(r-q, x)/np.dot(x, x)

def dist(p, q, r):
    ''' Helper function for closest_vertex
    Purpose: Calculate distance from vertex r to the line traced from p to q

    --- Parameters ---
    p : cord1 of pc [1x3]
    q : cord2 of pc [1x3]
    r : vertex [1x3]
    '''
    return np.linalg.norm(t(p, q, r)*(p-q)+q-r)

def closest_vertex(p, q, V):
    ''' Helper function for pca_vertex_association
    Purpose: Finds the closest vertex to given pc
    
    --- Parameters ---
    p : cord1 of pc [1x3]
    q : cord2 of pc [1x3]
    V : List of all vertices '''
    distances = [dist(p, q, r) for r in V]
    min_distance = min(distances)
    vertex_idx = distances.index(min_distance)    
    return V[vertex_idx], min_distance

def pca(H):
    ''' Principal Component Analysis
    Purpose: Find Principal Component of a cluster of hit points

    --- Parameters ---
    H : Hit points cluster [Nx3]

    --- Output ---
    Points p and q lying on principal component (each 1x3) '''
    
    pca = PCA(n_components=min(len(H), 3))
    pca.fit(H)
    pc_components = [pca.components_[:, i] for i in range(pca.components_.shape[1])]
    p = np.array([pca.mean_[i] for i in range(3)])
    pc_mag = pca.explained_variance_[0]
    if len(H) == 2:
        q = np.array([pca.mean_[0] + pc_components[0][0] * math.sqrt(pc_mag),
                      pca.mean_[1] + pc_components[1][0] * math.sqrt(pc_mag),
                      0])
    elif len(H) == 1:
        q = np.array([0, 0, 0])
    else:
        q = np.array([pca.mean_[0] + pc_components[0][0] * math.sqrt(pc_mag),
                      pca.mean_[1] + pc_components[1][0] * math.sqrt(pc_mag),
                      pca.mean_[2] + pc_components[2][0] * math.sqrt(pc_mag)])
    
    return p, q, pca.explained_variance_
     
def pca_vertex_association(H, V):
    ''' Main function
    Purpose: Associate a cluster with a vertex using pca
    
    --- Parameters ---
    H : Hit points cluster [Nx3]
    V : List of vertices
    
    --- Output ---
    Vertex (1x3) from list V, associated with cluster H'''

    p, q, strength = pca(H)
    vertex, min_distance = closest_vertex(p, q, V)
    return vertex, min_distance, strength
