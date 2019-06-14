import numpy as np

def energy_metrics(energies, predictions, labels):
    '''Calculates average clustering efficiency and purity. Using this
    function will be slightly more performant than calling 
    energy_efficiency and energy_purity individiually.

    Keyword arguments:
    energies -- (N,) numpy array of energies
    predictions -- (N,) numpy array of predictions
    labels -- (N,) numpy array of labels. Note that this function
           assumes that the prediction and label for a given voxel
           lie at the same position in these arrays.
    '''
    efficiencies = {label: [] for label in np.unique(labels)}
    purities = {label: [] for label in efficiencies.keys()}
    for label in np.unique(labels):
        correct_index = np.where(((predictions == label).astype('int') + (labels == label).astype('int')) == 2)
        correct_energy = np.sum(energies[correct_index])
        true_energy = np.sum(energies[np.where(labels == label)])
        predicted_energy = np.sum(energies[np.where(predictions == label)])
        efficiencies[label].append(correct_energy / true_energy)
        purities[label].append(correct_energy / predicted_energy)
    return efficiencies, purities

def energy_efficiency(energies, predictions, labels):
    '''Calculates average energy clustering efficiency given predictions and labels.

    Keyword arguments:
    energies -- (N,) numpy array of energies
    predictions -- (N,) numpy array of predictions
    labels -- (N,) numpy array of labels. Note that this function
           assumes that the prediction and label for a given voxel
           lie at the same position in these arrays.
    '''
    efficiencies = {label: [] for label in np.unique(labels)}
    for label in np.unique(labels):
        correct_index = np.where(((predictions == label).astype('int') + (labels == label).astype('int')) == 2)
        correct_energy = np.sum(energies[correct_index])
        true_energy = np.sum(energies[np.where(labels == label)])
        efficiencies[label].append(correct_energy / true_energy)
    return efficiencies

def energy_purity(energy, predictions, labels):
    '''Calculates average energy clustering purity given predictions and labels.
    
    Keyword arguments:
    energies -- (N,) numpy array of energies
    predictions -- (N,) numpy array of predictions
    labels -- (N,) numpy array of labels. Note that this function
           assumes that the prediction and label for a given voxel
           lie at the same position in these arrays.    
    '''
    purities = {label: [] for label in np.unique(labels)}
    for label in np.unique(labels):
        correct_index = np.where(((predictions == label).astype('int') + (labels == label).astype('int')) == 2)
        correct_energy = np.sum(energies[correct_index])
        predicted_energy = np.sum(energies[np.where(predictions == label)])
        purities[label].append(correct_energy / predicted_energy)
    return purities

def number_efficiency(predictions, labels):
    pass

def number_purity(predictions, labels):
    pass

if __name__ == '__main__':
    from hdf5_loader import load_and_offset_HDF5_to_sparse_np
    from clustering import group_clusters, simple_vertex_association
    efficiencies, purities = {0: [], 1: []}, {0: [], 1: []}
    files = ["192px_00.hdf5", "ArCube_0100.hdf5"]
    for q in files:
        big = 900 if q == files[0] else 9001
        for i in range(big):            
            d, c, f, l, o = load_and_offset_HDF5_to_sparse_np(q, 2, i*2)
            groups = group_clusters(np.vstack(c)[:, :3])
            predictions = simple_vertex_association(groups, o)
            energies = np.vstack(f)
            labels = np.vstack(c)[:, -1]
            e, p = energy_metrics(energies, predictions, labels)
            for j in [0, 1]:
                efficiencies[j].append(e[j])
                purities[j].append(p[j])
            if i % 10 == 0:
                print("%s - %d" % (q, i))
                c0e = np.nansum(efficiencies[0]) / np.count_nonzero(~np.isnan(efficiencies[0]))
                c1e = np.nansum(efficiencies[1]) / np.count_nonzero(~np.isnan(efficiencies[1]))
                c0p = np.nansum(purities[0]) / np.count_nonzero(~np.isnan(purities[0]))
                c1p = np.nansum(purities[1]) / np.count_nonzero(~np.isnan(purities[1]))
                ae, ap = (c0e + c1e)/2, (c0p + c1p) / 2
                print("\tavg Class 0 (e, p) = (%f, %f)" % (c0e, c0p))
                print("\tavg Class 1 (e, p) = (%f, %f)" % (c1e, c1p))
                print("\tavg total (e, p) = (%f, %f)" % (ae, ap))
