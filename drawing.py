import numpy as np
import plotly.graph_objs as go
from plotly.offline import plot
from pileup import shift_and_sum_events

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
    from hdf5_loader import load_and_convert_HDF5_to_sparse_np, linear_train_loader
    from clustering import dbscan
    if_filename = raw_input("\nWhat input file?\n")
    events = int(raw_input("How many events?\n"))
    epsilon = float(raw_input("Epsilon?\n"))
    min_samples = float(raw_input("Minimum samples?\n"))

    of_filename = 'graphs/%d-%d-%d.html' % (events, epsilon, min_samples)
    d, c, f, l, n = next(linear_train_loader(if_filename, events))
    _events = []
    ctr = 0
    for i in range(len(n)):
        _events.append(c[ctr:ctr+n[i], :])
        ctr += n[i]
    shifted_events = shift_and_sum_events(_events)
    clustering = dbscan(_events, epsilon, min_samples)
    signal = np.where(clustering.labels_ >= 0)
    c_ = (clustering.labels_ >= 0).astype(np.int32)
    c_[signal] = clustering.labels_[signal] + 1
    scatter(x=shifted_events[:, 0], y=shifted_events[:, 1], z=shifted_events[:, 2], color=c_, hovertext=['Cluster %d' % d for d in c_], filename=of_filename)
    print("Saved to " + of_filename)
