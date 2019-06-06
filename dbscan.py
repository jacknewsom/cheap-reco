from sklearn.cluster import DBSCAN

epsilon = 3
min_samples = 10

def dbscan(data, epsilon=epsilon, min_samples=min_samples):
    return DBSCAN(eps=epsilon, min_samples=min_samples, metric='euclidean').fit(data)
