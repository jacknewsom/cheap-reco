from sklearn.decomposition import PCA
import numpy as np
import math

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
    return V[vertex_idx]

def pca(H, V):
    ''' Principal Component Analysis
    Purpose: Find Principal Component of a cluster of hit points

    --- Parameters ---
    H : Hit points cluster [Nx3]
    V : List of all vertices 

    --- Output ---
    Points p and q lying on principal component (each 1x3) '''
    
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
    return p, q
    
    
def pca_vertex_association(H, V):
    ''' Main function
    Purpose: Associate a cluster with a vertex using pca
    
    --- Parameters ---
    H : Hit points cluster [Nx3]
    V : List of vertices
    
    --- Output ---
    Vertex (1x3) from list V, associated with cluster H'''

    p, q = pca(H)
    return closest_vertex(p, q, V)
    
