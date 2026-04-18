import numpy as np

def distance_euclidienne(v1, v2):
    return np.sqrt(np.sum((v1 - v2) ** 2))

def distance_canberra(v1, v2):
    return np.sum(np.abs(v1 - v2) / (np.abs(v1) + np.abs(v2) + 1e-8))

def distance_cosinus(v1, v2):
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    return 1 - (dot_product / (norm_v1 * norm_v2 + 1e-8))