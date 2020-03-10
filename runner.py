import h5py
import numpy as np
import scipy as sp
import argparse
from time import time
from plotly.offline import plot
from utils.drawing import scatter_hits, scatter_vertices, draw
from clustering_utils.clustering import cluster_and_cut
from clustering_utils.pca import pca, dist
from hough.hough import calculate_subclusters_with_grouping
from data_utils.event_generator import simulate_interaction


parser = argparse.ArgumentParser(description='Run reconstruction')
parser.add_argument('-n', '--nspills',
                    dest='n_spills',
                    type=int,
                    default=1,
                    help='number of spills')
parser.add_argument('--input_file',
                    dest='input_file',
                    default='jack.hdf5',
                    help='full path to input file')
parser.add_argument('--beam_intensity',
                    dest='beam_intensity',
                    default=1,
                    help='beam intensity in MW')
'''
parser.add_argument('--output_file',
                    dest='output_file',
                    default='reco.hdf5',
                    help='full path to output file')
'''
parser.add_argument('--seed',
                    dest='seed',
                    type=int,
                    default=12345,
                    help='random seed')
args = parser.parse_args()

np.random.seed(args.seed)

run_index = int(time())
print('Run index %d' % run_index)

current_event = 0
n_spills = args.n_spills
if n_spills < 0:
    n_spills = 1e8 # go until we run out of events in args.input_file

for event_index in range(n_spills):
    print('Analyzing event %d...' % event_index)

    data_load_start = time()
    events = []

    # approx. 124 events per spill per megawatt at 574m
    poisson_mean = 124 * args.beam_intensity
    n_events = np.random.poisson(poisson_mean)

    # load data into events
    while events is not None and len(events) == 0:
        events = simulate_interaction(args.input_file, n_events, current_event)
    if events is None:
        print('No more events in %s' % args.input_file)
        break

    # extract geometric and energy deposition data from each event
    current_event += n_events
    coordinates = [events[j]['coordinates'] for j in range(len(events))]
    features = [events[j]['energies'] for j in range(len(events))]
    vertices = [events[j]['vertex'] for j in range(len(events))]
    pdg_codes = [events[j]['pdg_codes'] for j in range(len(events))]
    kinetic_energies = [events[j]['kinetic_energies'] for j in range(len(events))]

    # define event label as index of event in events
    labels = [np.full((coordinates[j].shape[0], 1), j) if vertices[j] is not None else np.full((coordinates[j].shape[0], 1), -1) for j in range(len(events))]

    # reshape each element of features to be (n, 1)
    features = [f.reshape((-1, 1)) for f in features]

    # cut hits with less than 0.5MeV (about 78% of energy on average)
    # simple reconstruction appears hopeless otherwise, these hits
    # generally do not carry much useful information
    coordinates = np.vstack(coordinates)[:, :3]
    labels = np.vstack(labels)
    features = np.vstack(features)
    min_energy_idx = np.where(features > 0.5)[0]
    cut_energy = np.sum(features[min_energy_idx])
    coordinates = coordinates[min_energy_idx]
    labels = labels[min_energy_idx]
    features = features[min_energy_idx]

    # cluster data
    coordinates, labels, features, predictions = cluster_and_cut(
        coordinates,
        labels,
        features,
        0)
    
    # reshape lists into numpy arrays
    coordinates = np.vstack(coordinates)
    labels = np.hstack(labels)
    features = np.hstack(features)
    predictions = np.hstack(predictions)

    # create dictionary for accessing coordinates of only vertices within fiducial volume
    vertices_ = {j: vertices[j] for j in range(len(vertices)) if vertices[j] is not None}

    # reorganize data in dictionary where (key, value) = (cluster index, array of cluster hits)
    cluster_labels = np.unique(predictions)
    clusters = {}
    for cluster in cluster_labels:
        cluster_idx = np.where(predictions == cluster)[0]
        cluster_data = coordinates[cluster_idx]
        cluster_label = labels[cluster_idx]
        cluster_features = features[cluster_idx]
        clusters[cluster] = {
            'data': cluster_data,
            'label': cluster_label,
            'features': cluster_features}

    # Place each DBSCAN noise point in own cluster
    if -1 in clusters.keys():
        n_clusters = max(clusters.keys())
        for point in range(len(clusters[-1]["data"])):
            clusters[n_clusters+point+1] = {
                "data": clusters[-1]["data"][point].reshape((1,-1)),
                "label": np.array([clusters[-1]["label"][point]]),
                "features": np.array([clusters[-1]["features"][point]])}
        del clusters[-1]

    # set true_vertex to vertex with greatest associated energy
    for cluster in clusters:
        vertex_energy = {}
        for vertex in np.unique(clusters[cluster]['label']):
            vertex_idx = np.where(clusters[cluster]['label'] == vertex)[0]
            vertex_energy[vertex] = np.sum(clusters[cluster]['label'][vertex_idx])
        dominant_vertex = max(vertex_energy, key=vertex_energy.get)
        dominant_vertex_energy = vertex_energy[dominant_vertex] / np.sum(vertex_energy.values())
        clusters[cluster]['true_vertex'] = dominant_vertex
        clusters[cluster]['true_vertex_energy_fraction'] = dominant_vertex_energy

    # run PCA on clusters
    pca_start = time()
    non_None_vertices = [v for v in vertices if v is not None]
    bad_PCA = {}
    if non_None_vertices == []:
        for cluster in clusters:
            clusters[cluster]["prediction"] = -1
    else:
        cluster_keys = list(clusters.keys())
        for cluster in cluster_keys:
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
            if len(explained_variance) == 1:
                PCA_strength = 0
            elif explained_variance[1] == 0:
                PCA_strength = 0
            else:
                PCA_strength = explained_variance[0] / explained_variance[1]
            if PCA_strength >= 100 and min_dist <= 10:
                clusters[cluster]['prediction'] = min_vertex
            else:
                bad_PCA[cluster] = clusters[cluster]
                del clusters[cluster]
    print("\tPCA cluster association complete in %.3f[s]" % (time() - pca_start))

    # run baby Hough on clusters with poor PCA (i.e. clusters that fail PCA_strength >= 100 && min_dist <= 10)
    if bad_PCA != {}:
        hough_start = time()
        post_Hough = {}
        for cluster in bad_PCA:
            if bad_PCA[cluster]['true_vertex'] != -1:
                points = bad_PCA[cluster]['data']
                origin = vertices_[bad_PCA[cluster]['true_vertex']]
                subclusters, noise = calculate_subclusters_with_grouping(points, origin)
                label = bad_PCA[cluster]['label']
                features = bad_PCA[cluster]['features']
                for subcluster in subclusters:
                    if len(subcluster) == 0:
                        continue
                    points_, label_, features_ = points[subcluster], label[subcluster], features[subcluster]
                    post_Hough[len(post_Hough)] = {
                        'data': points_,
                        'label': label_,
                        'features': features_,
                        'true_vertex': bad_PCA[cluster]['true_vertex'],
                        'true_vertex_energy_fraction': bad_PCA[cluster]['true_vertex_energy_fraction'],
                    }
                if noise:
                    post_Hough[len(post_Hough)] = {
                        'data': points[noise],
                        'label': label[noise],
                        'features': features[noise],
                        'true_vertex': bad_PCA[cluster]['true_vertex'],
                        'true_vertex_energy_fraction': bad_PCA[cluster]['true_vertex_energy_fraction']
                    }
            else:
                post_Hough[len(post_Hough)] = bad_PCA[cluster]
        # run PCA again on these bad PCA clusters
        for cluster in post_Hough:
            p, q, explained_variance = pca(post_Hough[cluster]['data'])
            post_Hough[cluster]['PCA_explained_variance'] = explained_variance
            post_Hough[cluster]['vertices'] = {}

            min_dist = np.inf
            min_vertex = -1
            n_hits = len(post_Hough[cluster]['features'])
            for j in range(len(vertices)):
                post_Hough[cluster]['vertices'][j] = {}
                if vertices[j] is None:
                    continue
                post_Hough[cluster]['vertices'][j]['DOCA'] = dist(p, q, vertices[j])
                post_Hough[cluster]['vertices'][j]['distance_to_closest_point'] = np.amin(sp.spatial.distance_matrix(post_Hough[cluster]['data'], vertices[j].reshape((1, -1))))
                if post_Hough[cluster]['vertices'][j]['DOCA'] < min_dist:
                    min_dist = post_Hough[cluster]['vertices'][j]['DOCA']
                    min_vertex = j
            if len(explained_variance) == 1:
                PCA_strength = 0
            elif explained_variance[1] == 0:
                PCA_strength = 0
            else:
                PCA_strength = explained_variance[0] / explained_variance[1]
            post_Hough[cluster]['prediction'] = min_vertex
            '''
            # uncomment this block to enable cuts
            if PCA_strength >= 2 and min_dist <= 20 and n_hits > 5:
                post_Hough[cluster]['prediction'] = min_vertex
            else:
                # cluster failed PCA search even after Hough
                post_Hough[cluster]['prediction'] = -1
            '''
        clusters_ = {}
        for cluster in clusters:
            clusters_[len(clusters_)] = clusters[cluster]
        for cluster in post_Hough:
            clusters_[len(clusters_)] = post_Hough[cluster]

        clusters = clusters_
        print('\tHough fix complete in %.3f[s]' % (time() - hough_start))
    
    # plot some graphs
    graph_start = time()
    from hough.hough import cartesian_to_spherical as cts
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for cluster in clusters:
        if clusters[cluster]['true_vertex'] == -1 or clusters[cluster]['data'].shape[0] < 50:
            continue
        data = clusters[cluster]['data']
        ang = cts(data, vertices_[clusters[cluster]['true_vertex']])
        theta, phi = ang[:, 1], ang[:, 2]
        sub, noise = calculate_subclusters_with_grouping(data, vertices_[clusters[cluster]['true_vertex']])
        bus = {}
        data_, fdsa = [], []
        for idx in range(len(sub)):
            data_.append(data[sub[idx]])
            fdsa.append([idx] * len(sub[idx]))
        data_.append(data[noise])
        fdsa.append([-15] * len(noise))
        data = np.vstack(data_)
        fdsa = np.hstack(fdsa)
        x, y, z = data[:, 0], data[:, 1], data[:, 2]
        fig, ax = plt.subplots()
        plt.xlabel('Theta [deg]')
        plt.ylabel('Phi [deg]')
        plt.title('Run %d, Spill %d, Cluster %d, Num Sub %d' % (run_index, event_index, cluster, len(sub) + int(noise != [])))
        h = ax.hist2d(theta, phi, bins=[36, 18])
        plt.colorbar(h[3], ax=ax)
        plt.savefig('bh_plots/r%d-s%d-c%d.png' % (run_index, event_index, cluster))
        plt.close()
        
        vtx = vertices_[clusters[cluster]['true_vertex']]
        scatter = scatter_hits(x, y, z, fdsa, fdsa)
        vscatter = scatter_vertices([vtx], [0], [0])
        plot([scatter, vscatter], filename='bh_plots/r%d-s%d-c%d.html' % (run_index, event_index, cluster))
    print('\tGraph drawing complete in %.3f[s]' % (time()-graph_start))
        
    # Calculate accuracy
    correctly_labeled = 0
    total = 0
    for cluster in clusters:
        clusters[cluster]["all_vertices"] = vertices_
        if "vertices" not in clusters[cluster]:
            continue
        vertices_keys = clusters[cluster]["vertices"].keys()
        for vertex in vertices_keys:
            if clusters[cluster]["vertices"][vertex] == {}:
                del clusters[cluster]["vertices"][vertex]
        for label in clusters[cluster]['label']:
            total += 1
            if label == clusters[cluster]['prediction']:
                correctly_labeled += 1
    accuracy = float(correctly_labeled) / total

    draw_time = time()
    # remove 1 hit clusters
    clusters = {c: clusters[c] for c in clusters if clusters[c]['data'].shape != (1, 3)}
    x, y, z, pred, label = [], [], [], [], []
    colors = []
    color = 0
    for c in clusters:
        data = clusters[c]['data']
        prediction = clusters[c]['prediction']
        label_ = clusters[c]['true_vertex']
        x.append(data[:, 0])
        y.append(data[:, 1])
        z.append(data[:, 2])
        pred.append([prediction for j in range(len(x[-1]))])
        label.append([label_ for j in range(len(x[-1]))])
        if len(x[-1]) < 5:
            colors.append([-1 for j in range(len(x[-1]))])
        else:
            colors.append([color for j in range(len(x[-1]))])
            color += 1
    x = np.hstack(x)
    y = np.hstack(y)
    z = np.hstack(z)
    pred = np.hstack(pred)
    label = np.hstack(label)
    colors = np.hstack(colors)
    text = ['cluster:' + str(c) + ', pred:' + str(p) + ', true:' + str(l) for c, p, l in zip(colors, pred, label)]
    vertex_data = [(-1, np.array([0,0,0]))] + [(v, vertices_[v]) for v in vertices_]
    vcolors = [v[0] for v in vertex_data]
    vtext = ['Vertex %d' % v for v in vcolors]
    vhits = [v[1] for v in vertex_data]
    pred_hits = scatter_hits(x, y, z, pred, text)
    true_hits = scatter_hits(x, y, z, label, text)
    v_scatter = scatter_vertices(vhits, vcolors, vtext)
    filename = '3d_scatterplots/run-%d-spill-%d-' % (run_index, event_index)
    plot([pred_hits, v_scatter], filename=filename + 'pred.html')
    plot([true_hits, v_scatter], filename=filename + 'true.html')
    print("\tScatterplots drawn in %.3f[s]" % (time() - draw_time))
                
    write_time = time()
    output_file = 'reconstruction_output/reco-run%d-spill%d.hdf5' % (run_index, event_index)
    with h5py.File(output_file, "w") as f:
        for cluster in clusters:
            if 'PCA_explained_variance' not in clusters[cluster].keys():
                continue
            cluster_group = f.create_group("event-%d_cluster-%d" % (event_index, cluster))
            cluster_group.create_dataset("n_hits", data=clusters[cluster]["data"].shape[0])
            cluster_group.create_dataset("energy", data=np.sum(clusters[cluster]["features"]))
            cluster_group.create_dataset("PCA_component_strength", data=clusters[cluster]["PCA_explained_variance"], chunks=True)
            cluster_group.create_dataset("true_vertex", data=clusters[cluster]["true_vertex"])
            cluster_group.create_dataset("true_vertex_energy_fraction", data=clusters[cluster]["true_vertex_energy_fraction"])
            vertices = cluster_group.create_group("vertices")
            for vertex in vertices_:
                vertex_ = vertices.create_group("vertex-%d" % vertex)
                vertex_.create_dataset('DOCA', data=clusters[cluster]['vertices'][vertex]['DOCA'])
                vertex_.create_dataset('distance_to_closest_point', data=clusters[cluster]['vertices'][vertex]['distance_to_closest_point'])
    print("\tOutput saved to %s in %.3f[s]" % (output_file, time() - write_time))
