import numpy as np
from cad_similarity.similarity import correlation_similarity


def local_similarity(cluster_mats_a, cluster_mats_b):
    if not cluster_mats_a or not cluster_mats_b:
        return 0.0
    sims = []
    for a in cluster_mats_a:
        best = max(correlation_similarity(a, b) for b in cluster_mats_b)
        sims.append(best)
    return float(np.mean(sims))
