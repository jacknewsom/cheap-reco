import numpy as np
import plotly.graph_objs as go
from plotly.offline import plot

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

if __name__=='__main__':
    from hdf5_loader import load_and_convert_HDF5_to_sparse_np
    from dbscan import dbscan
    if_filename = raw_input("\nWhat input file?\n")
    events = int(raw_input("How many events?\n"))
    epsilon = float(raw_input("Epsilon?\n"))
    min_samples = float(raw_input("Minimum samples?\n"))
    of_filename = '%d-%d-%d.html' % (events, epsilon, min_samples)
    
    d, c, f, l = load_and_convert_HDF5_to_sparse_np(if_filename, 0, events)
    clustering = dbscan(c[:, :3], epsilon, min_samples)
    signal = np.where(clustering.labels_ >= 0)
    c_ = (clustering.labels_ >= 0).astype(np.int32)
    c_[signal] = clustering.labels_[signal] + 1
    scatter(x=c[:, 0], y=c[:, 1], z=c[:, 2], color=c_, hovertext=['Cluster %d' % d for d in c_], filename=of_filename)
    print("Saved to " + of_filename)
    
    
