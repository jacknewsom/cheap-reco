import numpy as np
from event_generator import simulate_interaction
from clustering import cluster_and_cut
from association import is_cluster_touching_vertex
from pca import pca_vertex_association
from drawing import draw_events_and_vertices as draw
from metrics import energy_accuracy, number_accuracy

e_accuracies = []
n_accuracies = []
for i in range(20):
    print("Analyzing event %d..." % i)
    # load data
    coordinates = []
    while len(coordinates) == 0:
        coordinates, features, _, vertices = simulate_interaction("data/ArCube_0000.hdf5")
    labels = [c[:, -1].reshape((-1, 1)) for c in coordinates]
    
    # cluster and cut data
    coordinates, labels, features, predictions = cluster_and_cut(np.vstack(coordinates)[:, :3], np.vstack(labels), np.vstack(features), 0)
    # just use large signal clusters
    coordinates, labels, features, predictions = coordinates[0], labels[0], features[0], predictions[0]

    print("\tData loaded")
    # split remaining data into two categories: vertex-touching and not
    cluster_labels = np.unique(predictions)
    clusters_touching_vertices = {}
    clusters_not_touching_vertices = {}
    for cluster in cluster_labels:
        cluster_data = coordinates[np.where(predictions == cluster)[0]]
        found_touching_vertex = False
        for j in range(len(vertices)):
            if is_cluster_touching_vertex(cluster_data, vertices[j]):
                clusters_touching_vertices[cluster] = {"data": cluster_data, "vertex": j}
                found_touching_vertex = True
                break
        if not found_touching_vertex:
            clusters_not_touching_vertices[cluster] = {"data": cluster_data}
    print("\tTouching cluster association complete")
    
    # run PCA on clusters that do not touch vertices directly
    for cluster in clusters_not_touching_vertices:
        closest_vertex = pca_vertex_association(clusters_not_touching_vertices[cluster]["data"], vertices)
        for j in range(len(vertices)):
            if np.all(closest_vertex == vertices[j]):
                clusters_not_touching_vertices[cluster]["vertex"] = j
        if "vertex" not in clusters_not_touching_vertices[cluster]:
            # something's wrong
            raise RuntimeError("PCA returned vertex not in `vertices`")
    print("\tPCA cluster association complete")
        
    # restructure data for graphing ease
    for group in [clusters_touching_vertices, clusters_not_touching_vertices]:
        for cluster in group:
            cluster_data = group[cluster]['data']
            cluster_vtx = group[cluster]['vertex']
            cluster_label = np.full((cluster_data.shape[0], 1), cluster_vtx)
            group[cluster]['data'] = np.hstack((cluster_data, cluster_label))

    clusters_touching_vertices = [clusters_touching_vertices[c]['data'] for c in clusters_touching_vertices]
    clusters_not_touching_vertices = [clusters_not_touching_vertices[c]['data'] for c in clusters_not_touching_vertices]
    if len(clusters_touching_vertices) == 0 and len(clusters_not_touching_vertices) != 0:
        all_clusters = np.vstack(clusters_not_touching_vertices)
    elif len(clusters_touching_vertices) != 0 and len(clusters_not_touching_vertices) == 0:
        all_clusters = np.vstack(clusters_touching_vertices)
    elif len(clusters_touching_vertices) == 0 and len(clusters_not_touching_vertices) == 0:
        # no clusters remaining after cut
        continue
    else:
        all_clusters = np.vstack([np.vstack(clusters_touching_vertices), np.vstack(clusters_not_touching_vertices)])

    # calculate accuracy
    data_dict = {}
    coords = np.vstack(coordinates)
    for j in range(coords.shape[0]):
        if tuple(coords[j]) not in data_dict:
            data_dict[tuple(coords[j])] = {}
        if tuple(all_clusters[j][:-1]) not in data_dict:
            data_dict[tuple(all_clusters[j][:-1])] = {}
        # label
        data_dict[tuple(coords[j])]["label"] = labels[j]
        data_dict[tuple(coords[j])]["energy"] = features[j]
        # prediction
        data_dict[tuple(all_clusters[j][:-1])]["prediction"] = all_clusters[j][-1]

    correct_energies = [data_dict[k]["energy"] for k in data_dict if data_dict[k]["prediction"] == data_dict[k]["label"]]
    e_accuracy = sum(correct_energies) / sum(features)
    as_list = [data_dict[k]["prediction"] == data_dict[k]["label"] for k in data_dict]
    n_accuracy = float(sum(as_list)) / len(as_list)
    print("\tEnergy Accuracy calculated: %.3f" % e_accuracy)
    print("\tNumber Accuracy calculated: %.3f" % n_accuracy)
    n_accuracies.append(n_accuracy)
    e_accuracies.append(e_accuracy)
    
    # draw graph!
    draw(coordinates, labels, labels, vertices, "drawings/%d-true.html" % i)
    draw(all_clusters[:, :3], all_clusters[:, -1], all_clusters[:, -1], vertices, "drawings/%d-pred-%.3f.html" % (i, e_accuracy))
    print("\tGraphs saved")

n_accuracy = float(sum(n_accuracies)) / len(n_accuracies)
e_accuracy = float(sum(e_accuracies)) / len(e_accuracies)
print("\n\nTotal Energy Accuracy: %.3f" % e_accuracy)
print("Total Number Accuracy: %.3f" % n_accuracy)
