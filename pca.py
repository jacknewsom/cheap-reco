from drawing import draw_events_and_vertices
from sklearn.decomposition import PCA
import plotly.graph_objs as go
from plotly.offline import plot
import numpy as np
import h5py
import os
import math

def get_graphs(input_filepath, output_dirpath, batch_size):
    ''' Create individual event graphs '''
    for event_idx in range(batch_size):
        d, c, f, l, v = loader(input_filepath, 1, event_idx)
        in_filename_ext = os.path.basename(in_filepath)
        in_filename_no_ext = os.path.splitext(in_filename_ext)[0]
        output_filepath = "{}{}_event_{}.html".format(output_dirpath,
                                                      in_filename_no_ext,
                                                      event_idx)
        draw_events_and_vertices(np.vstack(c)[:, :3], np.vstack(c)[:, -1],
                                 np.vstack(c)[:, -1], v, out_file)
        print("Saved: {}".format(output_filepath))
    
# Helper function for distance from point to line
def t(p, q, r):
    x = p-q
    return np.dot(r-q, x)/np.dot(x, x)

# Helper function for distance from point to line
def dist(p, q, r):
    return np.linalg.norm(t(p, q, r)*(p-q)+q-r)

# Find closest vertex
def estimate_vertex(p, q, V):
    '''H -- Hits, 
    p -- principal component coordinate 1
    q -- principal component coordinate 2
    V -- Vertices '''
    distance = []
    
    # Get the closest vertex
    distances = [np.linalg.norm(t(p, q, r)*(p-q)+q-r) for r in V]
    min_dist = min(distances)
    vertex_idx = distances.index(min_dist)
    
    return V[vertex_idx]


def pca(H, V):
    # Plot of all hit points
    hits_plot = go.Scatter3d(x=H[:, 0], y=H[:, 1], z=H[:, 2], mode='markers',
                             marker = dict(size=2))

    # Calculate Principal Components
    pca = PCA(n_components=3)
    pca.fit(H)
    pca_vx = pca.components_[:, 0]
    pca_vy = pca.components_[:, 1]
    pca_vz = pca.components_[:, 2]
    vector1_magnitude = pca.explained_variance_[0]
    vector2_magnitude = pca.explained_variance_[1]
    vector3_magnitude = pca.explained_variance_[2]
    
    # Plot first principal component
    pc1_x = [pca.mean_[0] + -pca_vx[0] * 5 * math.sqrt(vector1_magnitude),
             pca.mean_[0] + pca_vx[0] * 5 * math.sqrt(vector1_magnitude)]
    pc1_y = [pca.mean_[1] + -pca_vy[0] * 5 * math.sqrt(vector1_magnitude),
             pca.mean_[1] + pca_vy[0] * 5 * math.sqrt(vector1_magnitude)]
    pc1_z = [pca.mean_[2] + -pca_vz[0] * 5 * math.sqrt(vector1_magnitude),
             pca.mean_[2] + pca_vz[0] * 5 * math.sqrt(vector1_magnitude)]
    pc1_trace = go.Scatter3d(x=pc1_x, y=pc1_y, z=pc1_z, mode='lines',
                             line=dict(color='firebrick', width=4))

    # Get both endpoints of main principal component
    p = np.array([pca.mean_[0] + -pca_vx[0] * 2 * math.sqrt(vector1_magnitude),
                  pca.mean_[1] + -pca_vy[0] * 2 * math.sqrt(vector1_magnitude),
                  pca.mean_[2] + -pca_vz[0] * 2 * math.sqrt(vector1_magnitude)])
    q = np.array([pca.mean_[0] + pca_vx[0] * 2 * math.sqrt(vector1_magnitude),
                  pca.mean_[1] + pca_vy[0] * 2 * math.sqrt(vector1_magnitude),
                  pca.mean_[2] + pca_vz[0] * 2 * math.sqrt(vector1_magnitude)])

    # Calculate which vertex is closest
    vertex = estimate_vertex(p, q, V)
    
    # Plot vertex
    vertex_plot = go.Scatter3d(x=[vertex[0]], y=[vertex[1]],
                               z=[vertex[2]], mode='markers',
                               marker = dict(color='black', size=8))

    # Plot minimum distance from principal component to vertex
    r = vertex
    min_vertex_dist = t(p, q, r)*(p-q)+q
    min_x = min_vertex_dist[0]
    min_y = min_vertex_dist[1]
    min_z = min_vertex_dist[2]
#    import pdb
#    pdb.set_trace()
    distance_trace = go.Scatter3d(x=[min_x, r[0]], y=[min_y, r[1]], z=[min_z, r[2]], mode='lines')

    principal_components_trace2 = go.Scatter3d(x=[pca.mean_[0], min_x],
                                               y=[pca.mean_[1], min_y],
                                               z=[pca.mean_[2], min_z],
                                               mode='lines')

    # Create and save aggregate plot
    #fig = go.Figure(data=[hits_plot, pc1_trace, vertex_plot, distance_trace])
    #plot(fig, filename=output_filename)
    #print("\nDistance from point r to line is: {}\n".format(dist(p, q, r)))

    #return [hits_plot, pc1_trace, vertex_plot, distance_trace]
    return [hits_plot, principal_components_trace2, vertex_plot, distance_trace]

def hdf5_to_numpy(filename):
    with h5py.File(filename, 'r') as f:
        voxels = f['voxels']
        labels = f['labels']
        predictions = f['predictions']
        features = f['features']
        vertices = f['vertices']
        return voxels[:], labels[:], predictions[:], features[:], vertices[:]

def batch_pca(predictions, labels, voxels, vertices, output_filename):
    cluster_plots_list = []

    # First create a list of all the vertices
    vertex_list = []
    for cluster_label in np.unique(predictions):
        cluster_vertex_idx = labels[np.where(predictions == cluster_label)][0]
        vertex_list.append(vertices[cluster_vertex_idx])

    # Run pca and 
    for cluster_label in np.unique(predictions):
        cluster_idx = np.where(predictions == cluster_label)[0]
        cluster_voxels = voxels[cluster_idx]
        #cluster_vertex_idx = labels[np.where(predictions == cluster_label)][0]
        #cluster_vertex = vertices[cluster_vertex_idx]

        indiv_plots_list = pca(cluster_voxels, vertex_list)
        cluster_plots_list.append(indiv_plots_list[0])
        cluster_plots_list.append(indiv_plots_list[1])
        cluster_plots_list.append(indiv_plots_list[2])
        cluster_plots_list.append(indiv_plots_list[3])
        #for plot in indiv_plots_list:
        #    cluster_plots_list.append(plot)
            
    figure = go.Figure(data=cluster_plots_list)
    return plot(figure, filename=output_filename)

def hdf_loader(filename, event_idx):
    d, c, f, l, v = loader(filename, 1, event_idx)
    H = np.vstack(c)[:, :3]
    V = list(v[0])
    return H, V
    
if __name__ == "__main__":
    '''get_graphs(input_filepath="../ArCube_files/ArCube_0000.hdf5",
               output_dirpath="../ArCube_graphs/",
               batch_size=100)'''
    # event_idxs = [0, 18, 38, 41, 59, 60, 65, 88, 93, 94, 97]
    #  "../ArCube_files/ArCube_0000.hdf5", 1, event_idxs[1]
    
    #H, V = hdf_loader('../ArCube_clustered_hdfs/event0.hdf5', 1)
    #pca(H, V, "plotly_graph.html")

    voxels, labels, predictions, features, vertices = hdf5_to_numpy('../ArCube_clustered_hdfs/event0.hdf5')
    batch_pca(predictions, labels, voxels, vertices, "plotly_graph.html")
