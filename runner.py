import numpy as np
import scipy as sp
import scipy.spatial
import reconstruction.pca
from data_utils.event_generator import simulate_interaction
from reconstruction.clustering import cluster_and_cut
from reconstruction.association import is_cluster_touching_vertex_iterative, is_cluster_touching_vertex_clusterwise
from reconstruction.pca import pca_vertex_association
from utils.drawing import scatter_hits, scatter_vertices, draw
from utils.metrics import energy_accuracy, number_accuracy, energy_metrics
from time import time

e_accuracies = []
n_accuracies = []
e_efficiencies = []
e_purities = []
for i in range(50):
    print("Analyzing event %d..." % i)
    # load data
    data_load_start = time()
    events = []
    while len(events) == 0:
        events = simulate_interaction("jack.hdf5", 1)
    coordinates = [events[j]['coordinates'] for j in range(len(events))]
    features = [events[j]['energies'] for j in range(len(events))]
    vertices = [events[j]['vertex'] for j in range(len(events))]
    pdg_codes = [events[j]['pdg_codes'] for j in range(len(events))]
    kinetic_energies = [events[j]['kinetic_energies'] for j in range(len(events))]
    
    labels = [np.full((coordinates[j].shape[0], 1), j) if vertices[j] is not None else np.full((coordinates[j].shape[0], 1), -1) for j in range(len(coordinates))]
    features = [f.reshape((-1, 1)) for f in features]
    
    # cluster and cut data
    coordinates, labels, features, predictions = cluster_and_cut(np.vstack(coordinates)[:, :3], np.vstack(labels), np.vstack(features), 25)
    # separate small and large signal clusters
    small_coordinates, small_labels = coordinates[1], labels[1]
    small_features, small_predictions = features[1], predictions[1]
    coordinates, labels, features, predictions = coordinates[0], labels[0], features[0], predictions[0]
    print("\tData loaded in %.3f[s]" % (time() -  data_load_start))

    # split remaining data into two categories: vertex-touching and not
    touching_cluster_start = time()
    cluster_labels = np.unique(predictions)
    clusters_touching_vertices = {}
    clusters_not_touching_vertices = {}
    for cluster in cluster_labels:
        cluster_data = coordinates[np.where(predictions == cluster)[0]]
        found_touching_vertex = False
        touching_fn = is_cluster_touching_vertex_clusterwise if len(cluster_data) < 7500 else is_cluster_touching_vertex_iterative
        for j in range(len(vertices)):
            if vertices[j] is not None and touching_fn(cluster_data, vertices[j]):
                clusters_touching_vertices[cluster] = {"data": cluster_data, "vertex": j}
                found_touching_vertex = True
                break
        if not found_touching_vertex:
            clusters_not_touching_vertices[cluster] = {"data": cluster_data}
    print("\tTouching cluster association complete in %.3f[s]" % (time() - touching_cluster_start))
    
    # run PCA on clusters that do not touch vertices directly
    pca_start = time()
    unassociated_clusters = {}
    _clusters_not_touching_vertices = {}
    non_None_vertices = [v for v in vertices if v is not None]
    for cluster in clusters_not_touching_vertices:
        closest_vertex, distance = pca_vertex_association(clusters_not_touching_vertices[cluster]["data"], non_None_vertices)
        # ignore clusters more than minimum distance away from nearest vertex
        if distance >= reconstruction.pca.cutoff_distance:
            unassociated_clusters[cluster] = {"data": clusters_not_touching_vertices[cluster]["data"]}
            continue
        for j in range(len(vertices)):
            if np.all(closest_vertex == vertices[j]):
                _clusters_not_touching_vertices[cluster] = {"data": clusters_not_touching_vertices[cluster]["data"], "vertex": j}
        if "vertex" not in _clusters_not_touching_vertices[cluster]:
            # something's wrong
            raise RuntimeError("PCA returned vertex not in `vertices`")
    clusters_not_touching_vertices = _clusters_not_touching_vertices
    print("\tPCA cluster association complete in %.3f[s]" % (time() - pca_start))

    # associate small clusters with same vertex as nearest big cluster
    small_cluster_start = time()
    small_clusters = {}
    cluster_labels = np.unique(small_predictions)
    for small_cluster in cluster_labels:
        cluster_data = small_coordinates[np.where(small_predictions == small_cluster)[0]]
        mean = np.mean(cluster_data, axis=0).reshape((1, -1))
        closest_vertex, closest_distance = None, np.inf
        if clusters_touching_vertices == {} and clusters_not_touching_vertices == {}:
            dist_matrix = sp.spatial.distance_matrix(mean, np.vstack(vertices))
            closest_vertex = np.argmin(dist_matrix)
        else:
            for group in [clusters_touching_vertices, clusters_not_touching_vertices]:
                for big_cluster in group:
                    dist_matrix = sp.spatial.distance_matrix(mean, group[big_cluster]['data'])
                    group_closest_distance = np.amin(dist_matrix)
                    if closest_distance > group_closest_distance:
                        closest_distance = group_closest_distance
                        closest_vertex = group[big_cluster]['vertex']
        small_clusters[small_cluster] = {}
        small_clusters[small_cluster]['data'] = cluster_data
        small_clusters[small_cluster]['vertex'] = closest_vertex
    print("\tSmall cluster nearest-neighbor association complete in %.3f[s]" % (time() - small_cluster_start))
        
    # restructure data for graphing ease
    for group in [clusters_touching_vertices, clusters_not_touching_vertices, small_clusters]:
        for cluster in group:
            cluster_data = group[cluster]['data']
            cluster_vtx = group[cluster]['vertex']
            cluster_label = np.full((cluster_data.shape[0], 1), cluster_vtx)
            group[cluster]['data'] = np.hstack((cluster_data, cluster_label))

    # add small clusters to clusters_not_touching_vertices
    for small_cluster in small_clusters:
        clusters_not_touching_vertices[small_cluster] = small_clusters[small_cluster]

    clusters_touching_vertices = [clusters_touching_vertices[c]['data'] for c in clusters_touching_vertices]
    clusters_not_touching_vertices = [clusters_not_touching_vertices[c]['data'] for c in clusters_not_touching_vertices]
    unassociated_clusters = [unassociated_clusters[c]['data'] for c in unassociated_clusters]
    if len(clusters_touching_vertices) == 0 and len(clusters_not_touching_vertices) != 0:
        all_assoc_clusters = np.vstack(clusters_not_touching_vertices)
    elif len(clusters_touching_vertices) != 0 and len(clusters_not_touching_vertices) == 0:
        all_assoc_clusters = np.vstack(clusters_touching_vertices)
    elif len(clusters_touching_vertices) == 0 and len(clusters_not_touching_vertices) == 0:
        # no clusters remaining after cut
        continue
    else:
        all_assoc_clusters = np.vstack([np.vstack(clusters_touching_vertices), np.vstack(clusters_not_touching_vertices)])

    # calculate accuracy
    if len(unassociated_clusters) > 0:
        unassociated_clusters = np.vstack(unassociated_clusters)
        unassoc_label = np.full((unassociated_clusters.shape[0], 1), -1)
        unassoc_with_label = np.hstack((unassociated_clusters, unassoc_label))
        all_clusters = np.vstack((all_assoc_clusters, unassoc_with_label))
    else:
        all_clusters = all_assoc_clusters
        
    data_dict = {}
    coords = np.vstack((small_coordinates, coordinates))
    feats = np.hstack((small_features, features))
    labs = np.hstack((small_labels, labels))
    for j in range(coords.shape[0]):
        if tuple(coords[j]) not in data_dict:
            data_dict[tuple(coords[j])] = {}
        if tuple(all_clusters[j][:-1]) not in data_dict:
            data_dict[tuple(all_clusters[j][:-1])] = {}
        # label
        data_dict[tuple(coords[j])]["label"] = labs[j]
        data_dict[tuple(coords[j])]["energy"] = feats[j]
        # prediction
        data_dict[tuple(all_clusters[j][:-1])]["prediction"] = all_clusters[j][-1]

    correct_energies = [data_dict[k]["energy"] for k in data_dict if data_dict[k]["prediction"] == data_dict[k]["label"]]
    e_accuracy = sum(correct_energies) / sum(feats)
    as_list = [data_dict[k]["prediction"] == data_dict[k]["label"] for k in data_dict]
    n_accuracy = float(sum(as_list)) / len(as_list)
    n_accuracies.append(n_accuracy)
    e_accuracies.append(e_accuracy)
    print("\tEnergy Accuracy calculated: %.3f" % e_accuracy)
    print("\tNumber Accuracy calculated: %.3f" % n_accuracy)
    
    _energies = np.array([data_dict[k]["energy"] for k in data_dict])
    _predictions = np.array([data_dict[k]["prediction"] for k in data_dict])
    _labels = np.array([data_dict[k]["label"] for k in data_dict])
    efficiency, purity = energy_metrics(_energies, _predictions, _labels)
    efficiency = np.nansum(efficiency.values()) / np.sum(~np.isnan(efficiency.values()))
    purity = np.nansum(purity.values()) / np.sum(~np.isnan(purity.values()))
    e_efficiencies.append(efficiency)
    e_purities.append(purity)
    print("\tEnergy Clustering Efficiency calculated: %.3f" % efficiency)
    print("\tEnergy Clustering Purity calculated: %.3f" % purity)

    # fix color bug by adding a point under each vertex at the vertex's location
    drawing_start = time()
    for j in range(len(vertices)):
        if vertices[j] is None:
            continue
        new_point = np.hstack((vertices[j], j))
        coords = np.vstack((coords, vertices[j]))
        labels = np.hstack((labels, j))
        all_assoc_clusters = np.vstack((all_assoc_clusters, new_point))
        
    # draw graph!
    '''
    true_clusters_scatterplot = scatter_hits(coords[:, 0],
                                             coords[:, 1],
                                             coords[:, 2],
                                             labs)
    associated_clusters_scatterplot = scatter_hits(all_assoc_clusters[:, 0],
                                                   all_assoc_clusters[:, 1],
                                                   all_assoc_clusters[:, 2],
                                                   all_assoc_clusters[:, 3])
    vertex_scatterplot = scatter_vertices(non_None_vertices)
    draw("drawings/%d-true.html" % i, true_clusters_scatterplot, vertex_scatterplot)    
    if len(unassociated_clusters) > 0:
        unassociated_clusters_scatterplot = scatter_hits(unassociated_clusters[:, 0],
                                                         unassociated_clusters[:, 1],
                                                         unassociated_clusters[:, 2],
                                                         [-1 for k in range(len(unassociated_clusters))])
        draw("drawings/%d-pred-%.3f.html" % (i, e_accuracy),
             associated_clusters_scatterplot,
             unassociated_clusters_scatterplot,
             vertex_scatterplot)
    else:
        draw("drawings/%d-pred-%.3f.html" % (i, e_accuracy),
             associated_clusters_scatterplot,
             vertex_scatterplot)
    '''
    scatterplots = []
    # separate fiducial truth and non-fiducial truth
    fiducial_c, nonfiducial_c = [], []
    fiducial_l, nonfiducial_l = [], []
    for coordinate, label in zip(coords, labs):
        if label == -1:
            nonfiducial_c.append(coordinate)
            nonfiducial_l.append(label)
        else:
            fiducial_c.append(coordinate)
            fiducial_l.append(label)
    for true_c, true_l in zip([fiducial_c, nonfiducial_c], [fiducial_l, nonfiducial_l]):
        if true_c != []:
            true_c = np.vstack(true_c)
            true_l = np.vstack(true_l)
            scatterplots.append(scatter_hits(true_c[:, 0],
                                             true_c[:, 1],
                                             true_c[:, 2],
                                             true_l))
    # separate associated and unassociated prediction
    
            
    
    print("\tGraphs saved in %.3f[s]" % (time() - drawing_start))
    print("\tTotal time elapsed: %.3f[s]" % (time() - data_load_start))

n_accuracy = float(sum(n_accuracies)) / len(n_accuracies)
e_accuracy = float(sum(e_accuracies)) / len(e_accuracies)
e_efficiency = float(sum(e_efficiencies)) / len(e_efficiencies)
e_purity = float(sum(e_purities)) / len(e_purities)
print("\n\nTotal Energy Accuracy: %.3f" % e_accuracy)
print("Total Number Accuracy: %.3f" % n_accuracy)
print("Total Energy Efficiency: %.3f" % e_efficiency)
print("Total Energy Purity: %.3f" % e_purity)
