import numpy as np

def unit_vector(vector):
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    return np.arccos(np.clip(np.dot(unit_vector(v1), unit_vector(v2)), -1.0, 1.0))