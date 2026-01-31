import numpy as np
from sklearn.cluster import DBSCAN


def dbscan_clusters(points, eps=0.05, min_samples=10):
    points = np.asarray(points, dtype=float)
    model = DBSCAN(eps=eps, min_samples=min_samples)
    return model.fit_predict(points)
