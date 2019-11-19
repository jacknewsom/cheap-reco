import numpy as np
import plotly.graph_objs as go
from plotly.offline import plot

def scatter_hits(x, y, z, colors, event_text=None):
    '''Returns plotly Scatter3d object formatted properly
    for hit data.

    Keyword arguments:
    x -- (N,) numpy array of x-coordinates
    y -- (N,) numpy array of y-coordinates
    z -- (N,) numpy array of z-coordinates
    '''
    return go.Scatter3d(x=x, y=y, z=z, mode='markers', name=None,
                          marker=dict(
                              size=2,
                              color=colors,
                              colorscale=None,
                              opacity=0.7
                          ),
                          hoverinfo=['x', 'y', 'z'] if event_text is None else ['x', 'y', 'z', 'text'],
                          hovertext=event_text
                          )

def scatter_vertices(vertices, vcolors, vtext=None):
    '''Returns plotly Scatter3d object formatted properly
    for vertex data.
    
    Keyword arguments:
    vertices -- (M,) list of (1, 3) numpy arrays
    '''
    vertices = np.vstack(vertices)
    vx, vy, vz = vertices[:, 0], vertices[:, 1], vertices[:, 2]
    return go.Scatter3d(x=vx, y=vy, z=vz, mode='markers', name=None,
                      marker=dict(
                          size=10,
                          color=vcolors,
                          colorscale=None,
                          opacity=0.9,
                      ),
                      hoverinfo=['x', 'y', 'z'] if vtext is None else ['x', 'y', 'z', 'text'],
                      hovertext=vtext
                      )

def draw(filename, *scatterplots):
    '''Take list of events, vertices, and draw both
    '''
    scatters = [s for s in scatterplots]
    return plot(scatters, filename=filename)
