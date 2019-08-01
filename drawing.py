import numpy as np
import plotly.graph_objs as go
from plotly.offline import plot
from hdf5_loader import load_and_offset_HDF5_to_sparse_np

def scatter(x, y, z, color, colorscale=None, markersize=2, name=None, hovertext=None, filename='temp.html'):
    '''Draw 3D scatterplot.

    Keyword arguments:
    x -- (N,) numpy array of x-coordinates
    y -- (N,) numpy array of y-coordinates
    z -- (N,) numpy array of z-coordinates
    color -- (N,) array-like object of colors for each of N points
    colorscale -- maps 'color' argument to actual color of points in scatterplot
    markersize -- width of points in scatterplot
    name -- name of dataset
    hovertext -- (N,) array-like object of labels for each point in scatterplot
    filename -- name of output file
    '''
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

def draw_events_and_vertices(events, event_colors, event_text, vertices, filename):
    '''Draw 3D scatterplot of some number of events and their respective vertices.

    Keyword arguments:
    events -- (N, 3) numpy array of coordinates
    event_colors -- (N,) array-like object of colors for each of N events
    event_text -- (N,) array-like object of labels for each point in events
    vertices -- (M,) list of (1, 3) numpy arrays
    filename -- name of output file
    '''
    x, y, z = events[:, 0], events[:, 1], events[:, 2]
    threed = go.Scatter3d(x=x, y=y, z=z, mode='markers', name=None,
                          marker=dict(
                              size=2,
                              color=event_colors,
                              colorscale=None,
                              opacity=0.7
                          ),
                          hoverinfo=['x', 'y', 'z'] if event_text is None else ['x', 'y', 'z', 'text'],
                          hovertext=event_text
                          )
    vertices = np.vstack(vertices)
    vx, vy, vz = vertices[:, 0], vertices[:, 1], vertices[:, 2]
    vcolors = list(range(vertices.shape[0]))
    vtext = ['Vertex %d' % d for d in range(vertices.shape[0])]
    vd = go.Scatter3d(x=vx, y=vy, z=vz, mode='markers', name=None,
                      marker=dict(
                          size=10,
                          color=vcolors,
                          colorscale=None,
                          opacity=0.9,
                      ),
                      hoverinfo=['x', 'y', 'z'] if vtext is None else ['x', 'y', 'z', 'text'],
                      hovertext=vtext
                      )
    return plot([threed, vd], filename=filename)
