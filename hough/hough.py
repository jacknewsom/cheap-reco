import numpy as np

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
        theta_bin, phi_bin = min(int(point[1]/10), 35), min(int(point[2]/10), 17)
        bins[theta_bin, phi_bin] += 1
    return bins

def calculate_subclusters(points, origin):
    ''' Takes (x, y, z) data and determines subclusters
    based on angular distribution of data

    returns 
    '''
    angular = cartesian_to_spherical(points, origin)
    bins = histogram_angular_distribution(angular)
    clusters_idx = {}
    noise_idx = []
    idx = 0
    for cartesian, spherical in zip(points, angular):
        theta, phi = min(int(spherical[1]/10), 35), min(int(spherical[2]/10), 17)
        if bins[theta, phi] > 10:
            if (theta, phi) not in clusters_idx.keys():
                clusters_idx[(theta, phi)] = []
            clusters_idx[(theta, phi)].append(idx)
        else:
            noise_idx.append(idx)
        idx += 1
    return clusters_idx.values(), noise_idx

def calculate_subclusters_with_grouping(points, origin, threshold_factor=0.1):
    ''' Takes (x, y, z) data and determines subclusters
    based on angular distribution of data, then groups data
    close together in (theta, phi) space
    '''
    def depth_first_search(arr, i, j, label):
        if arr[i, j] != -1:
            return
        elif bins[i, j] < noise_threshold:
            return
        arr[i, j] = label
        for del_i in [-1, 0, 1]:
            for del_j in [-1, 0, 1]:
                if del_i == 0 and del_j == 0:
                    continue
                m, n = i+del_i, j+del_j
                if m >= arr.shape[0]:
                    m = 0
                elif m < -arr.shape[0]:
                    m = 1
                if n >= arr.shape[1]:
                    n = 0
                elif n < -arr.shape[1]:
                    n = 1
                depth_first_search(arr, m, n, label)

    angular = cartesian_to_spherical(points, origin)
    bins = histogram_angular_distribution(angular)
    mean = np.average(bins)
    stdev = np.std(bins)

    noise_idx = []
    cluster_num = 0
    cluster_labels = np.zeros_like(bins) - 1
    # label all points
    while -1 in cluster_labels:
        i, j = np.unravel_index(np.argmax(bins), bins.shape)
        peak = bins[i, j]
        if peak == 0:
            break
        # bins closer than this distance to the mean are considered noise
        noise_threshold = threshold_factor * peak
        depth_first_search(cluster_labels, i, j, cluster_num)
        bins[i, j] = 0
        cluster_num += 1
        
    # collect points into clusters
    clusters_idx = {i: [] for i in range(cluster_num+1)}
    for i, spherical in enumerate(angular):
        theta, phi = min(int(spherical[1]/10), 35), min(int(spherical[2]/10), 17)
        cluster_label = cluster_labels[theta, phi]
        if cluster_label == -1:
            noise_idx.append(i)
        else:
            clusters_idx[cluster_label].append(i)
    keys = list(clusters_idx.keys())
    for key in keys:
        if len(clusters_idx[key]) == 0:
            del clusters_idx[key]
    return clusters_idx.values(), noise_idx
        
    
