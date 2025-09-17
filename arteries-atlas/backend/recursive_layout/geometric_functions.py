import numpy as np

from ..utils.numpy_utils import orthogonal, scalar


def paths_collisions(path1, path2):
    """Compute all the intersection occurrences between two trajectories.
    """

    def ccw(A, B, C):
        return (C[..., 1] - A[..., 1]) * (B[..., 0] - A[..., 0]) > (B[..., 1] - A[..., 1]) * (C[..., 0] - A[..., 0])

    A, B = path1[:-1, None, :], path1[1:, None, :]
    C, D = path2[None, :-1, :], path2[None, 1:, :]

    notparallels = scalar(B - A, D - C) ** 2 < scalar(B - A, B - A) * scalar(D - C, D - C) - 1e-8

    return np.logical_and(notparallels, np.logical_and(ccw(A, C, D) != ccw(B, C, D), ccw(A, B, C) != ccw(A, B, D)))


def multiple_paths_collisions(paths):
    concat_trajs = np.concatenate(paths)

    intersects = np.triu(paths_collisions(concat_trajs, concat_trajs), k=2)

    cumsize = np.concatenate([[0], np.cumsum([t.shape[0] for t in paths])])
    transitions = cumsize[np.logical_and(cumsize > 0, cumsize <= intersects.shape[0])] - 1

    intersects[transitions, :] = False
    intersects[:, transitions] = False

    collisions = np.argwhere(intersects)

    if collisions.size > 0:
        collision_trajectories = np.argmax(cumsize[None, :, None] > collisions[:, None, :], axis=1) - 1
        collision_indices = collisions - cumsize[collision_trajectories]
        return collision_trajectories, collision_indices
    else:
        return np.zeros((0, 2)), np.zeros((0, 2))


def path_semiline_collisions(paths, origin, vect, maxval=1e5):
    semiline = np.stack([origin, origin + maxval * vect], axis=0)
    return paths_collisions(paths, semiline)


def path_line_collisions(path, origin, vect):
    orth_vect = orthogonal(vect)
    proj_point, proj_traj = scalar(origin, orth_vect), scalar(path, orth_vect)

    return (proj_traj[:-1] - proj_point) * (proj_traj[1:] - proj_point) < 0


def path_semiplane_collisions(path, point, vect):
    proj_point, proj_traj = scalar(point, vect), scalar(path, vect)
    return (proj_traj[:-1] - proj_point) > 0


def path_insidebox_collisions(path, bounding_box):
    anchors = bounding_box.anchors
    orientation_vector = bounding_box.orientation_vector()
    bound_vectors = bounding_box.border_vectors()

    return np.logical_and(path_semiplane_collisions(path, anchors[0], orientation_vector),
                          path_semiplane_collisions(path, anchors[0], orthogonal(bound_vectors[0])),
                          path_semiplane_collisions(path, anchors[1], orthogonal(bound_vectors[1], "right")))


"""def cone_intersections(traj, anchors, angles):
    vect0, vect1 = angle_to_vect(angles[0]), angle_to_vect(angles[1])

    intersections = np.logical_or(
        path_line_collisions(traj, anchors[0], vect0),
        path_line_collisions(traj, anchors[1], vect1)
    )
    intersections[1:] = np.logical_or(intersections[1:],
                                      np.any(paths_collisions(traj[1:], np.array([anchors[0], anchors[1]])),
                                             axis=-1))
    return intersections"""

"""def cone_intersections2(traj, point1, point2, angles): # TODO supprimer
    vect0, vect1 = angle_to_vect(angles[0]), angle_to_vect(angles[1])
    return np.logical_or(path_line_collisions(traj, point1, vect0), path_line_collisions(traj, point2, vect1))"""
