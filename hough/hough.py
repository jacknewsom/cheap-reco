import numpy as np

points = np.array([
    [0,0,0],
    [1,1,1],
    [1,-1,1],
    [5,9,7],
])
origin = np.zeros((1,3))

def cartesian_to_spherical(points, origin):
    ''' Converts (x, y, z) data to (r, theta, phi) data
    where theta is azithumal angle and angular data is in
    degrees
    '''
    shift = points - origin
    angular = np.zeros(shift.shape)
    xy_2 = shift[:, 0] ** 2 + shift[:, 1] ** 2
    # r = sqrt(x^2 + y^2 + x^2)
    angular[:, 0] = np.sqrt(xy_2 + shift[:, 2] ** 2)
    # theta = arctan(y/x)
    angular[:, 1] = np.arctan2(shift[:, 1], shift[:, 0]) * (180 / np.pi) + 90
    # phi = arccos(z / r) = arctan(z / sqrt(x^2 + y^2))
    angular[:, 2] = np.arctan2(np.sqrt(xy_2), shift[:, 2]) * (180 / np.pi) + 90
    return angular

def histogram_angular_distribution(angular):
    ''' Bins (r, theta, phi) points based on (theta, phi)
    into 36 x 18 histogram (since 0 <= theta <= 2pi, 0 <= phi <= pi)
    '''
    bins = np.zeros((36, 18))
    for point in angular:
        theta_bin, phi_bin = int(point[1]/10), int(point[2]/10)
        bins[theta_bin, phi_bin] += 1
    return bins

def calculate_subclusters(points, origin):
    ''' Takes (x, y, z) data and determines subclusters
    based on angular distribution of data

    returns 
    '''
    angular = cartesian_to_spherical(points, origin)
    bins = histogram_angular_distribution(angular)
    background = np.average(bins)
    clusters = {}
    for cartesian, spherical in zip(points, angular):
        theta, phi = int(spherical[1]/10), int(spherical[2]/10)
        if bins[theta, phi] > 10:
            if (theta, phi) not in clusters.keys():
                clusters[(theta, phi)] = []
            clusters[(theta, phi)].append(cartesian)
    for cluster in clusters:
        clusters[cluster] = np.vstack(clusters[cluster])
    return clusters.values()
