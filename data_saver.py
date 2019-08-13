import numpy as np
import h5py
from event_generator import simulate_interaction
from clustering import cluster_and_cut
from drawing import draw_events_and_vertices as draw

for i in range(10):
    print("Event %d" % i)
    coordinates, features, _, vertices = simulate_interaction('data/ArCube_0000.hdf5')
    while len(coordinates) == 0:
        coordinates, features, _, vertices = simulate_interaction('data/ArCube_0000.hdf5')
    labels = [c[:, -1].reshape((-1, 1)) for c in coordinates]
    cd, cl, cf, cp = cluster_and_cut(np.vstack(coordinates)[:, :3], np.vstack(labels), np.vstack(features), 20)
    draw(cd, cl, cl, vertices, "events/%d-dn-true.html" % i)
    draw(cd, cp, cp, vertices, "events/%d-dn-pred.html" % i)
    draw(np.vstack(coordinates), np.vstack(labels).reshape(-1), np.vstack(labels).reshape(-1), vertices, "events/%d-true.html" % i)
    n_points = len(cl)
    with h5py.File('events/event{}.hdf5'.format(i), 'w') as f:        
        voxels = f.create_dataset('voxels', (n_points, 3), dtype=np.dtype('int32'))
        labels_ = f.create_dataset('labels', (n_points, 1), dtype=np.dtype('int32'))
        predictions_ = f.create_dataset('predictions', (n_points, 1), dtype=np.dtype('int32'))
        features_ = f.create_dataset('features', (n_points, 1), dtype=np.dtype('float32'))
        vertices_ = f.create_dataset('vertices', (len(vertices), 3), dtype=np.dtype('int32'))
        for j in range(len(vertices)):
            vertices_[j] = vertices[j]
        for j in range(n_points):
            voxels[j] = cd[j]
            labels_[j] = cl[j]
            predictions_[j] = cp[j]
            features_[j] = cf[j]
    print("\tfinished")
