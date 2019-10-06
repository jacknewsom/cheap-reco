import numpy as np
import scipy as sp
import os
import h5py
import argparse
import scipy.spatial
import reconstruction.pca
from data_utils.event_generator import simulate_interaction
from reconstruction.clustering import cluster_and_cut
from reconstruction.association import is_cluster_touching_vertex_iterative, is_cluster_touching_vertex_clusterwise
from reconstruction.pca import pca_vertex_association, pca, dist
from utils.metrics import energy_accuracy, number_accuracy, energy_metrics
from time import time


parser = argparse.ArgumentParser(description="Run reconstruction")
parser.add_argument('--n', '--nspills', dest='nspills', type=int, default=1000, help='number of spills')
parser.add_argument('--input_file', dest='input_file', default='jack.hdf5', help='full path to input file')
parser.add_argument('--beam_intensity', dest='beam_intensity', default=1, help='beam intensity in MW')
args = parser.parse_args()

e_accuracies = []
n_accuracies = []
e_efficiencies = []
e_purities = []
correct_dist_strength_pairs = []
incorrect_dist_strength_pairs = []

if not os.path.isdir('reconstruction_output'):
    os.mkdir('reconstruction_output')
    
run_index = int(time())
print("Run index %d" % run_index)
current_event = 0
for i in range(args.nspills):
    print("Analyzing event %d..." % i)
    # load data
    data_load_start = time()
    events = []

    # approximately 124 events per spill per megawatt at 574m
    poisson_mean = 124 * args.beam_intensity
    n_events = np.random.poisson(poisson_mean)

    while events is not None and len(events) == 0:
        events = simulate_interaction(args.input_file, n_events, current_event)
    if events is None:
        print "Ran out of events in this file"
        break
    current_event += n_events
    coordinates = [events[j]['coordinates'] for j in range(len(events))]
    features = [events[j]['energies'] for j in range(len(events))]
    vertices = [events[j]['vertex'] for j in range(len(events))]
    pdg_codes = [events[j]['pdg_codes'] for j in range(len(events))]
    kinetic_energies = [events[j]['kinetic_energies'] for j in range(len(events))]
    
    labels = [np.full((coordinates[j].shape[0], 1), j) if vertices[j] is not None else np.full((coordinates[j].shape[0], 1), -1) for j in range(len(coordinates))]
    features = [f.reshape((-1, 1)) for f in features]
    
    # cluster and cut data
    coordinates, labels, features, predictions = cluster_and_cut(np.vstack(coordinates)[:, :3], np.vstack(labels), np.vstack(features), 0)
    coordinates, labels, features, predictions = np.vstack(coordinates), np.hstack(labels), np.hstack(features), np.hstack(predictions)

    # cut hits with less than 0.5MeV
    min_energy_idx = np.where(features > 0.5)
    coordinates, labels, features, predictions = coordinates[min_energy_idx], labels[min_energy_idx], features[min_energy_idx], predictions[min_energy_idx]
    print("\tData loaded in %.3f[s]" % (time() -  data_load_start))

    # Organize data into clusters
    cluster_labels = np.unique(predictions)
    clusters = {}
    for cluster in cluster_labels:
        cluster_idx = np.where(predictions == cluster)[0]
        cluster_data = coordinates[cluster_idx]
        cluster_label = labels[cluster_idx]
        cluster_features = features[cluster_idx]
        clusters[cluster] = {"data": cluster_data, "label": cluster_label, "features": cluster_features}

    # Place each DBSCAN noise point in own cluster
    if -1 in clusters.keys():
        n_clusters = max(clusters.keys())
        for point in range(len(clusters[-1]["data"])):
            clusters[n_clusters+point+1] = {"data": clusters[-1]["data"][point].reshape((1,-1)), "label": [clusters[-1]["label"][point]], "features": [clusters[-1]["features"][point]]}
        del clusters[-1]
            
    # run PCA on clusters
    pca_start = time()
    non_None_vertices = [v for v in vertices if v is not None]
    if non_None_vertices == []:
        for cluster in clusters:
            clusters[cluster]["prediction"] = -1
    else:
        for cluster in clusters:
            p, q, explained_variance = pca(clusters[cluster]["data"])
            clusters[cluster]["PCA_explained_variance"] = explained_variance
            clusters[cluster]["vertices"] = {}
            # for each vertex, save DOCA and distance to closest point in cluster
            min_dist = np.inf
            min_vertex = -1
            for j in range(len(vertices)):
                clusters[cluster]["vertices"][j] = {}
                if vertices[j] is None:
                    continue
                clusters[cluster]["vertices"][j]["DOCA"] = dist(p, q, vertices[j])
                clusters[cluster]["vertices"][j]["distance_to_closest_point"] = np.amin(sp.spatial.distance_matrix(clusters[cluster]["data"], vertices[j].reshape((1, -1))))
                if clusters[cluster]["vertices"][j]["DOCA"] < min_dist:
                    min_dist = clusters[cluster]["vertices"][j]["DOCA"]
                    min_vertex = j
            clusters[cluster]["prediction"] = min_vertex
    print("\tPCA cluster association complete in %.3f[s]" % (time() - pca_start))

    # Fix cut vertices problem
    vertices_ = {j: vertices[j] for j in range(len(vertices)) if vertices[j] is not None}
    
    # Calculate accuracy
    correctly_labeled = 0
    for cluster in clusters:
        clusters[cluster]["all_vertices"] = vertices_
        if "vertices" not in clusters[cluster]:
            continue
        vertices_keys = clusters[cluster]["vertices"].keys()
        for vertex in vertices_keys:
            if clusters[cluster]["vertices"][vertex] == {}:
                del clusters[cluster]["vertices"][vertex]
        for label in clusters[cluster]['label']:
            if label == clusters[cluster]['prediction']:
                correctly_labeled += 1        

    write_time = time()
    with h5py.File("reconstruction_output/run-%d.hdf5" % run_index, "w") as f:
        for cluster in clusters:
            cluster_group = f.create_group("event-%d_cluster-%d" % (i, cluster))
            cluster_group.create_dataset("n_hits", data=clusters[cluster]["data"].shape[0])
            cluster_group.create_dataset("energy", data=np.sum(clusters[cluster]["features"]))
            cluster_group.create_dataset("PCA_component_strength", data=clusters[cluster]["PCA_explained_variance"], chunks=True)
            cluster_group.create_dataset("true_vertex", data=clusters[cluster]["label"], chunks=True)
            vertices = cluster_group.create_group("vertices")
            for vertex in vertices_:
                vertex_ = vertices.create_group("vertex-%d" % vertex)
                vertex_.create_dataset('coordinates', data=vertices_[vertex])
                vertex_.create_dataset('DOCA', data=clusters[cluster]['vertices'][vertex]['DOCA'])
                vertex_.create_dataset('distance_to_closest_point', data=clusters[cluster]['vertices'][vertex]['distance_to_closest_point'])
    print("\tOutput saved to reconstruction_output/run-%d.hdf5 in %.3f[s]" % (run_index, time() - write_time))
    print("\tTotal time elapsed %.3f[s]" % (time() - data_load_start))
