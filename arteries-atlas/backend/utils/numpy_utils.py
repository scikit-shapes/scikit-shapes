# ruff: noqa: EXE002

import numpy as np


def sort(indices, array):
    return indices[np.argsort(array[indices])]


def order(array): # TODO changer nom
    return np.argsort(np.argsort(array, axis=0).squeeze())


def integrate(y0, y):
    u = (y0 + np.cumsum(y, axis=0))

    if u.ndim == 1:
        u = u.reshape(-1, 1)

    return u


def integrate_angular_path(lengths, angles, initial_position=0):
    return integrate(initial_position, lengths * angle_to_vect(angles.squeeze()))


def partition_cone(angles, proportions): # TODO supprimer
    bounds = np.cumsum(proportions)
    return angles[0] + (angles[1] - angles[0]) * np.concatenate([[0], bounds / bounds[-1]])


def angle_to_vect(angle):
    angle = np.array(angle)
    return np.stack([np.cos(angle), np.sin(angle)], axis=angle.ndim)


def orthogonal(vect, orientation="left"):
    if orientation == "left":
        return np.stack([- vect[..., 1], vect[..., 0]], axis=vect.ndim - 1)
    else:
        return np.stack([vect[..., 1], - vect[..., 0]], axis=vect.ndim - 1)


def scalar(v, w):
    return (v * w).sum(-1)

def lower_projection(point1, point2, v):
    return scalar(point2 - point1, v) > 0

def orthogonal_projection(point, origin, vect):
    vect_normalized = vect / np.linalg.norm(vect) if np.linalg.norm(vect) > 0 else 0

    return origin + scalar(point - origin, vect_normalized) * vect_normalized


def projection(point, direction, origin, vect):
    orth_direction = orthogonal(direction)
    vect_normalized = vect / scalar(vect, orth_direction) if scalar(vect, orth_direction) != 0 else 0

    return origin + scalar((point - origin), orth_direction) * vect_normalized
