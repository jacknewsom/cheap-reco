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
