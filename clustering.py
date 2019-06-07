import numpy as np
import scipy as sp
from sklearn.cluster import DBSCAN

class Vertex:
    def __init__(self, number):
        self.id = number
        self.points = None
        self.neighbors = []
        self.color = None
    
    def add_neighbor(self, neighbor):
        assert isinstance(neighbor, Vertex)
        if neighbor not in self.neighbors and neighbor != self:
            self.neighbors.append(neighbor)

    def __repr__(self):
        return 'Vertex %d' % self.id

class Color:
    def __init__(self):
        self.value = None

    def __repr__(self):
        return str(self.value)

def dbscan(data, epsilon=3, min_samples=10):
    return DBSCAN(eps=epsilon, min_samples=min_samples, metric='euclidean').fit(data)

def closest_clusters(clusters):
    '''Determines which clusters should be grouped together.
    '''
    closest_cluster = np.full((len(clusters)), -1)
    for i, c in enumerate(clusters):
        clusters_ = clusters[:i] + clusters[i+1:]
        min_cluster, min_dist = -1, np.inf
        for j, c_ in enumerate(clusters_):
            dist = np.amin(sp.spatial.distance.cdist(c, c_))
            if dist < min_dist:
                min_cluster, min_dist = j, dist
        if min_cluster >= i:
            min_cluster += 1
        closest_cluster[i] = min_cluster
    assert len(clusters) == 1 or -1 not in closest_cluster
    return list(closest_cluster)

def find_vertex_in_graph(vertex_id, graph):
    '''Returns Vertex object with id vertex_id if such a
    Vertex exists in graph. Otherwise, creates a Vertex
    with vertex_id, adds it to graph, and returns it.
    '''
    for i in range(len(graph)):
        if graph[i].id == vertex_id:
            return graph[i]
    v = Vertex(vertex_id)
    graph.append(v)
    return v

def construct_graph(clusters):
    '''Takes in a list of clusters and returns a
    list composed of Vertex objects that represent
    each cluster.
    '''
    closest_cluster = closest_clusters(clusters)
    if len(closest_cluster) == 1:
        vertex = Vertex(-1)
        vertex.points = np.vstack(clusters)
        return [vertex]
    graph = []
    for v_i, v_f in enumerate(closest_cluster):
        vertex_i = find_vertex_in_graph(v_i, graph)
        vertex_i.points = clusters[v_i]
        vertex_f = find_vertex_in_graph(v_f, graph)
        vertex_f.points = clusters[v_f]
        vertex_i.add_neighbor(vertex_f)
        vertex_f.add_neighbor(vertex_i)
    return graph

def color_vertex(vertex, color=None):
    '''Takes in a Vertex object and a color
    and recursively colors Vertex and its
    neighbors.
    '''
    if vertex.color != None:
        return vertex.color
    elif vertex.color == None and color == None:
        color = Color()
    vertex.color = color
    for neighbor in vertex.neighbors:
        color_vertex(neighbor, color)
    return color

def color_graph(graph):
    '''Takes in a list of Vertex objects, mutates
    these elements such that all Vertex objects
    in a clique have the same color, then returns
    the list of colors.
    '''
    colors = []
    for vertex in graph:
        if vertex.color == None:
            colors.append(color_vertex(vertex))
    for i, color in enumerate(colors):
        color.value = i
    return colors

def color_interaction(data, epsilon=3, min_samples=10):
    '''Takes an NxM array of points. Performs DBSCAN on
    the points to form clusters. Then, connects each cluster
    to its nearest neighbor and colors all connected clusters.
    Returns data (not in same order) plus an extra column for
    color (i.e. an Nx(M+1) array).
    '''
    labels = dbscan(data, epsilon, min_samples).labels_
    signal_indices = np.where(labels!=-1)
    noise_indices = np.where(labels==1)
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
    colors = color_graph(graph)
    colored_signal = []
    for cluster in graph:
        color = cluster.color.value
        colored_signal.append(np.hstack((cluster.points, np.full((cluster.points.shape[0], 1), color))))
    colored_noise = np.hstack((noise, np.full((noise.shape[0], 1), -1)))
    colored_signal = np.vstack(colored_signal)
    return np.vstack([colored_noise, colored_signal])

if __name__ == '__main__':
    clusters =[np.array([[1, 2],
       [3, 4],
       [5, 6]]), np.array([[2, 4],
       [5, 1],
       [6, 2]]), np.array([[1, 1],
       [2, 5],
       [7, 0]])]
    c1, c2, c3 = [clusters[i] for i in range(3)]
    
