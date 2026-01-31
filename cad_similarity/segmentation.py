import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier


def build_features(points, normals, k=8):
    points = np.asarray(points, dtype=float)
    normals = np.asarray(normals, dtype=float)
    r = np.linalg.norm(points, axis=1, keepdims=True)

    nbrs = NearestNeighbors(n_neighbors=min(k + 1, len(points)))
    nbrs.fit(points)
    dists, _ = nbrs.kneighbors(points)
    mean_knn = dists[:, 1:].mean(axis=1, keepdims=True)

    X = np.hstack([points, normals, r, mean_knn])
    return X


def train_models(X, y, seed=0):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=int)

    dt = DecisionTreeClassifier(random_state=seed)
    dt.fit(X, y)

    mlp = MLPClassifier(
        hidden_layer_sizes=(50, 20),
        random_state=seed,
        max_iter=200,
        solver="lbfgs",
    )
    mlp.fit(X, y)

    return {"decision_tree": dt, "mlp": mlp}
