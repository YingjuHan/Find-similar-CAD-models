import numpy as np
from cad_similarity.segmentation import build_features, train_models

def test_feature_shape_and_train_models():
    rng = np.random.default_rng(0)
    points = rng.normal(size=(30, 3))
    normals = rng.normal(size=(30, 3))
    X = build_features(points, normals, k=3)
    assert X.shape[0] == 30
    y = np.array([0, 1] * 15)
    models = train_models(X, y, seed=0)
    assert "decision_tree" in models
    assert "mlp" in models
    pred = models["decision_tree"].predict(X[:5])
    assert pred.shape == (5,)
