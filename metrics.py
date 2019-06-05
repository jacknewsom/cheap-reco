import numpy as np

def efficiency(energies, predictions, labels):
    '''Calculates efficiency from sparse matrices energies, predictions,
    and labels, each of size (N, 4).
    '''
    intersection = np.unique(np.vstack((predictions[:, :3], labels[:, :3])))
    return np.sum(energies[intersection]) / np.sum(energies[labels[:, :3]])

def purity(energies, predictions, labels):
    '''Calculates purity from sparse matrices energies, predictions,
    and labels, each of size (N, 4).
    '''
    intersection = np.unique(np.vstack((predictions[:, :3], labels[:, :3])))
    return np.sum(energies[intersection]) / np.sum(energies[predictions[:, :3]])

def accuracy(predictions, labels):
    '''Calculates total segmentation accuracy from sparse matrices
    predictions and labels, both of size (N, 4)
    '''
    assert predictions.shape == labels.shape
    return np.sum(predictions[:, 3] == labels[:, 3]) / float(labels.shape[0])

