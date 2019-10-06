import numpy as np
import h5py

'''This file contains HDF5_to_dense_np and HDF5_to_sparse_np which
load data from HDF5 into numpy, but in different ways.

HDF5_to_standard_np loads HDF5 data into the 'dense' representation
for a tensor. It returns a full-size tensor with an entry in every
place of the tensor, even if the entry is 0.

HDF5_to_sparse_np loads HDF5 data into the sparse  representation
for a tensor. Instead of returning a full-size tensor, the function
returns the dimensions of the tensor and a list of only the nonzero
entries of the tensor and the corresponding values of those entries.
'''

def HDF5_data_dimension(filename):
    '''Returns size of data stored in HDF5 file.
    '''
    with h5py.File(filename, 'r') as f:
        return f['dimension'][0]

def load_and_convert_HDF5_to_dense_np(filename, start_index, batch_size):
    '''Load HDF5 data into a numpy tensor in dense representation.

    Keyword arguments:
    filename -- HDF5 file to load from
    start_index -- first instance number to load
    batch_size -- number of instances to convert to dense
                  numpy arrays
    '''
    with h5py.File(filename, 'r') as f:
        dim = f['dimension'][0]
        voxels_x = f['voxels_x']
        voxels_y = f['voxels_y']
        voxels_z = f['voxels_z']

        energies = f['energies']
        _labels = f['labels']

        data = np.empty((batch_size, dim, dim, dim))
        labels = np.empty((batch_size, dim, dim, dim))
        for i in range(start_index, start_index+batch_size):
            x, y, z = [j[i].astype(np.int) for j in [voxels_x, voxels_y, voxels_z]]
            data[i-start_index, x, y, z] = energies[i]
            labels[i-start_index, x, y, z] = _labels[i]
    return data, labels
        
def load_and_convert_HDF5_to_sparse_np(filename, batch_size, start_index=0):
    '''Load HDF5 data from multiple events into sparse tensors, shift the events
    by various offsets, then add the events together

    Keyword arguments:
    filename -- name of input HDF5 file
    batch_size -- number of events to load
    start_index -- location to start loading events from
    '''
    c, f, l, v = [], [], [], []
    loader = linear_train_loader(filename, 1, start_index)
    counter = 0
    
    while counter < batch_size:
        d, c_, f_, l_, _, v_ = next(loader)
        c_[:, -1] = counter
        c.append(c_)
        f.append(f_)
        l.append(l_)
        v.append(v_)
        counter += 1
    return d, c, f, l, v

def linear_train_loader(filename, batch_size, start_from=0):
    '''Generator that yields numpy data converted from HDF5 batchwise.

    Keyword arguments:
    filename -- name of input HDF5 file
    batch_size -- number of events to load
    start_index -- location to start loading events from
    '''
    with h5py.File(filename, 'r') as f:
        dim = f['dimension'][0]
        voxels_x = f['voxels_x']
        voxels_y = f['voxels_y']
        voxels_z = f['voxels_z']
        energies = f['energies']
        _labels = f['labels']
        
        for start_index in range(start_from,  10000, batch_size):
            num_entries_per_event = [len(voxels_x[i]) for i in range(start_index, start_index+batch_size)]
            total_entries = sum(num_entries_per_event)
            coordinates = np.empty((total_entries, 4))
            features = np.empty((total_entries, 1))
            labels = np.empty((total_entries, 1))

            c_ind = 0

            for i in range(start_index, start_index+batch_size):
                end = c_ind+num_entries_per_event[i-start_index]
                coordinates[c_ind: end, 0] = voxels_x[i]
                coordinates[c_ind: end, 1] = voxels_y[i]
                coordinates[c_ind: end, 2] = voxels_z[i]
                coordinates[c_ind: end, 3] = i-start_index
                
                features[c_ind: end] = energies[i].reshape((energies[i].shape[0], 1))
                labels[c_ind: end] = _labels[i].reshape((_labels[i].shape[0], 1))

                c_ind += len(voxels_x[i])

            if 'vertex' in f.keys():
                yield dim, coordinates, features, labels, num_entries_per_event, tuple(f['vertex'][start_index])
            else:
                yield dim, coordinates, features, labels, num_entries_per_event, (0,0,0)

def load_HDF5_from_dataset_keys(filename, keys, batch_size, start_from=0):
    with h5py.File(filename, 'r') as f:
        data = {key: [] for key in keys}
        for i in range(start_from, start_from+batch_size):
            # make sure this index exists, i.e. there are enough events
            if len(f['voxels_x']) < i: # we've run out of events
                return [None for key in keys]
            for key in data:
                if key == 'coordinates':
                    coordinates = [f['voxels_%s' % j][i] for j in ['x', 'y', 'z']]
                    coordinates = [c.reshape((-1, 1)) for c in coordinates]
                    coordinates = np.hstack(coordinates)
                    data[key].append(coordinates)
                else:
                    data[key].append(f[key][i])
        return [data[key] for key in keys]
    
def load_output_HDF5_to_dict(filename):
    with h5py.File(filename, 'r') as f:
        clusters = {}
        for cluster_key in f.keys():
            cluster_idx = len(clusters)
            clusters[cluster_idx] = {}
            clusters[cluster_idx]['PCA_component_strength'] = f[cluster_key]['PCA_component_strength'].__array__()
            clusters[cluster_idx]['energy'] = f[cluster_key]['energy'][()]
            clusters[cluster_idx]['n_hits'] = f[cluster_key]['n_hits'][()]
            clusters[cluster_idx]['true_vertex'] = f[cluster_key]['true_vertex'][()]
            clusters[cluster_idx]['vertices'] = {}
            for vertex_key in f[cluster_key]['vertices'].keys():
                clusters[cluster_idx]['vertices'][vertex_key] = {}
                clusters[cluster_idx]['vertices'][vertex_key]['DOCA'] = f[cluster_key]['vertices'][vertex_key]['DOCA'][()]
                clusters[cluster_idx]['vertices'][vertex_key]['distance_to_closest_point'] = f[cluster_key]['vertices'][vertex_key]['distance_to_closest_point'][()]
    return clusters

        
        
