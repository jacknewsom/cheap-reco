import os
os.sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt
from data_utils.hdf5_loader import load_output_HDF5_to_dict

def load_reconstruction_data(data_directory):
    if data_directory[-1] != '/':
        data_directory += '/'
    hdf5_files = [f for f in os.listdir(data_directory) if f[-5:] == '.hdf5']
    for file_ in hdf5_files:
        yield load_output_HDF5_to_dict(data_directory + file_)

def extract_interesting_bits(clusters, min_size, max_size):
    def cluster_prediction(cluster):
        min_key, min_DOCA = -1, float('inf')
        for vertex in cluster['vertices']:
            if cluster['vertices'][vertex]['DOCA'] < min_DOCA:
                min_key, min_DOCA = vertex, cluster['vertices'][vertex]['DOCA']
        return int(min_key.split('-')[-1])
    true = {}
    false = {}
    for key in ['dist', 'DOCA', 'PCA', 'Nhits']:
        true[key], false[key] = [], []
    for key in clusters:
        prediction = cluster_prediction(clusters[key])
        true_vertex_idx = clusters[key]['true_vertex']
        if clusters[key]['n_hits'] < min_size or clusters[key]['n_hits'] > max_size:
            continue
        for vertex in clusters[key]['vertices']:
            dist = clusters[key]['vertices'][vertex]['distance_to_closest_point']
            DOCA = clusters[key]['vertices'][vertex]['DOCA']
            PCA_component_strength = clusters[key]['PCA_component_strength']
            if len(PCA_component_strength) == 1 or PCA_component_strength[1] == 0:
                PCA_first_component_strength = 0
            else:
                PCA_first_component_strength = PCA_component_strength[0] / PCA_component_strength[1]
            Nhits = clusters[key]['n_hits']
            if prediction == true_vertex_idx:
                true['dist'].append(dist)
                true['DOCA'].append(DOCA)
                true['PCA'].append(PCA_first_component_strength)
                true['Nhits'].append(Nhits)
            else:
                false['dist'].append(dist)
                false['DOCA'].append(DOCA)
                false['PCA'].append(PCA_first_component_strength)
                false['Nhits'].append(Nhits)
    return true, false
                                        
def plot_X_vs_Y(X, Y, filename=None, bins=None):
    if bins:
        plt.hist2d(X, Y, bins=bins)
    else:
        plt.hist2d(X, Y)
    plt.savefig(filename)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Draw histograms")
    parser.add_argument('--l', dest='l', type=int, default=0, help='Lower bound of cluster size range')
    parser.add_argument('--u', dest='u', type=int, default=int(1e6), help='Upper bound of cluster size range')
    args = parser.parse_args()
    
    print("Scraping together the interesting stuff...")
    keys = ['dist', 'DOCA', 'PCA', 'Nhits']
    pairs = {0:
             {'x': 'dist', 'xlabel': 'Distance to Closest Voxel',
              'y': 'DOCA', 'ylabel': 'DOCA'},
             1:
             {'x': 'PCA', 'xlabel': 'Relative First Principal Component Magnitude',
              'y': 'DOCA', 'ylabel': 'DOCA'},
             2:
             {'x': 'Nhits', 'xlabel': 'Number of Hits in Cluster',
              'y': 'dist', 'ylabel': 'Distance to Closest Voxel'}}
    
    true, false = {key: [] for key in keys}, {key: [] for key in keys}
    for clusters in load_reconstruction_data("../reconstruction_output"):
        print("Loading another file with %d clusters" % len(clusters))
        t, f = extract_interesting_bits(clusters, args.l, args.u)
        for key in keys:
            true[key] += t[key]
            false[key] += f[key]
    for data in range(2):
        for pair in pairs:
            end_bit = 'Inc' if data == 1 else 'C'
            end_bit += 'orrectly Labeled Clusters'
            filename = 'true' if data == 1 else 'false'
            data_ = data
            data = [true, false][data]
            if args.u < 1e6 and args.l > 0:
                filename += '_%s-%s-%d-%d.png' % (pairs[pair]['x'], pairs[pair]['y'], args.l, args.u)
            else:
                filename += '_%s-%s' % (pairs[pair]['x'], pairs[pair]['y'])
            xlabel, ylabel = pairs[pair]['xlabel'], pairs[pair]['ylabel']
            fig, ax = plt.subplots()
            h = ax.hist2d(data[pairs[pair]['x']], data[pairs[pair]['y']], bins=100, cmap='inferno')
            plt.colorbar(h[3], ax=ax)
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            plt.title(xlabel + ' vs. ' + ylabel + ' For ' + end_bit)
            plt.savefig(filename)
            data = data_
