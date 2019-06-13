import numpy as np
import plotly.graph_objs as go
from plotly.offline import plot
from hdf5_loader import load_and_offset_HDF5_to_sparse_np
from clustering import color_interaction, braindead_vertex_association

def scatter(x, y, z, color, colorscale=None, markersize=2, name=None, hovertext=None, filename='temp.html'):
    threed = go.Scatter3d(x=x, y=y, z=z, mode='markers', name=name,
                      marker = dict(
                          size=markersize,
                          color=color,
                          colorscale=colorscale,
                          opacity=0.7
                      ),
                      hoverinfo=['x', 'y', 'z'] if hovertext is None else ['x', 'y','z', 'text'],
                      hovertext=hovertext
                      )
    return plot([threed], filename=filename)

def plot_a_bunch_of_events(filename, n_files=10, events_per_file=2):
    for i in range(n_files):
        print("file %d" % i)
        index = np.random.random_integers(0, 500)
        d, c, f, l, o = load_and_offset_HDF5_to_sparse_np(filename, events_per_file, index)
        print("\tloaded file")
        data = np.vstack(c)
        # import code
        # code.interact(local=locals())
        clusters = color_interaction(data[:, :3], 2, 2)
        print("\tclustered")
        predictions = braindead_vertex_association(clusters, o)
        print("\tpredicted")
        clusters_with_pred = []
        j = 0
        for label, cluster in zip(predictions, clusters):
            print("\tcluster %d" % j)
            clusters_with_pred.append(np.hstack((cluster, np.full((cluster.shape[0], 1), label))))
            j += 1
        p = np.vstack(clusters_with_pred)
        c = np.vstack(c)
        scatter(x=c[:, 0], y=c[:, 1], z=c[:, 2], color=c[:, 3], hovertext=['True Event %d' % d for d in c[:, 3]], filename=filename.split('.')[0] + '-true-%d-%d-%d.html' % (n_files, events_per_file, i))
        scatter(x=p[:, 0], y=p[:, 1], z=p[:, 2], color=p[:, 3], hovertext=['Predicted Event %d' %d for d in p[:, 3]], filename=filename.split('.')[0] + '-pred-%d-%d-%d.html' % (n_files, events_per_file, i))
        
            
            
            
