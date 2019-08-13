import numpy as np
from event_generator import simulate_interaction
from clustering import cluster_and_cut
from association import is_cluster_touching_vertex

for i in range(100):
    # load data
    coordinates = []
    while len(coordinates) == 0:
        coordinates, features, _, vertices = simulate_interaction("data/ArCube_0000.hdf5")
    labels = [c[:, -1].reshape((-1, 1)) for c in coordinates]

    # cluster and cut data
    coordinates, labels, features, predictions = cluster_and_cut(np.vstack(coordinates)[:, :3], np.vstack(labels), np.vstack(features), 25)

    # split remaining data into two categories: vertex-touching and not
    cluster_labels = np.unique(predictions)
    clusters_touching_vertices = []
    clusters_not_touching_vertices = []
    for cluster in cluster_labels:
        cluster_data = coordinates[np.where(predictions == cluster)[0]]
        found_touching_vertex = False
        for j in range(len(vertices)):
            if is_cluster_touching_vertex(cluster_data, vertices[j]):
                clusters_touching_vertices.append((cluster_data, cluster, j))
                found_touching_vertex = True
                break
        if not found_touching_vertex:
            clusters_not_touching_vertices.append((cluster_data, cluster))

    # run PCA on clusters that do not touch vertices directly
    
    
            
    
    
