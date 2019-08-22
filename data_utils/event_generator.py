import numpy as np
from hdf5_loader import load_and_convert_HDF5_to_sparse_np, load_HDF5_from_dataset_keys

def simulate_interaction(event_file, beam_intensity):
    '''Simulates neutrino interaction events in large
    volume of liquid argon.

    Event vertices are assumed to be distributed over
    a 7m x 3m x 5m volume. Then, two cuts are performed.
    First, events with vertices outside the fiducial volume
    are removed. Then, events without muons with kinetic energy
    above 500 MeV as primaries are removed.

    Keyword Arguments:
    event_file -- string, HDF5 file to load events from
    beam_intensity -- float, beam intensity in megawatts
    '''
    def vertex_in_fiducial_volume(vertex):
        x = abs(vertex[0]) < 300
        y = abs(vertex[1]) < 100
        z = 50 < vertex[2] and vertex[2] < 350
        return x and y and z

    def contains_energetic_muon(pdg_codes, kinetic_energies):
        return True in (kinetic_energies[pdg_codes == 13] > 500)
    
    # approximately 124 events per spill per megawatt at 574m
    poisson_mean = 124 * beam_intensity
    n_events = np.random.poisson(poisson_mean)
    
    # load n_events
    start_index = np.random.randint(0, 9999-n_events)
    keys = ['coordinates', 'energies', 'vertex', 'pdg_codes', 'kinetic_energies']
    c, e, v, pdg, ke = load_HDF5_from_dataset_keys(event_file, keys, n_events, start_index)
    events = []
    for i in range(n_events):
        # reject events with no hits in detector volume
        if len(c[i]) == 0:
            continue
        # reject event if no energetic muon
        if not contains_energetic_muon(pdg[i], ke[i]):
            continue
        events.append({})
        
        events[-1]['coordinates'] = c[i]
        events[-1]['energies'] = e[i]

        # reject vertex if outside fiducial volume
        if not vertex_in_fiducial_volume(v[i]):
            events[-1]['vertex'] = None
        else:
            events[-1]['vertex'] = v[i]

        events[-1]['pdg_codes'] = pdg[i]
        events[-1]['kinetic_energies'] = ke[i]
    return events
        
def simulate_interaction_without_pileup(event_file):
    '''Simulates neutrino interaction events in a large
    volume of Argon, then returns contents of a smaller
    volume.

    The volume events are distributed over is
    7m x 4m x 4m (= 112m^3). The smaller volume
    is a 2m x 2m x 2m (=8m^3) box within.

    The number density per cubic meter of vertices
    is Poisson distributed with mean 0.25 vertices/m^3.
    Thus, on average, 28 (112 * 0.25) vertices will be placed
    in the larger volume. 
    
    Keyword Arguments:
    event_file -- string, file to read events from
    '''
    def voxel_in_detector_box(voxel):
        xy = lambda c: c >= 333 and c < 1000
        z = lambda c: c >= 1333 and c < 2000
        return xy(voxel[0]) and xy(voxel[1]) and z(voxel[2])
        
    # determine number of vertices in larger volume
    n_events = np.random.poisson(28)

    # load n_events
    start_index = np.random.randint(0, 999-n_events)
    d, c, f, l, v = load_and_convert_HDF5_to_sparse_np(event_file, n_events, start_index)
    
    # distribute events across larger volume with
    # a uniform random distribution
    x_positions = np.random.randint(0, 1333, n_events)
    y_positions = np.random.randint(0, 1333, n_events)
    z_positions = np.random.randint(0, 2333, n_events)
    positions = [np.array([x, y, z]) for x, y, z in zip(x_positions, y_positions, z_positions)]
    shifts = [vert - p for vert, p in zip(v, positions)]
    vertices = positions

    # slice out smaller detector volume by rejecting
    # coordinates outside the box
    legal_coordinates, legal_vertices = [], []
    legal_features, legal_labels = [], []
    for event, shift, vertex, feature, label in zip(c, shifts, vertices, f, l):
        # for now, ignore events with vertices outside volume
        if not voxel_in_detector_box(vertex):
            continue
        new_event = event[:, :3] - shift
        legal_voxels = np.apply_along_axis(voxel_in_detector_box, 1, new_event)
        # if no voxels in detector volume, reject event and vertex
        if np.all(~legal_voxels):
            continue
        new_event = new_event[legal_voxels]
        new_event = np.hstack((new_event, np.full((new_event.shape[0], 1), len(legal_coordinates))))
        new_label = label[legal_voxels]
        new_feature = feature[legal_voxels]
        legal_coordinates.append(new_event)
        legal_vertices.append(vertex)
        legal_features.append(new_feature)
        legal_labels.append(new_label)
    
    return legal_coordinates, legal_features, legal_labels, legal_vertices
