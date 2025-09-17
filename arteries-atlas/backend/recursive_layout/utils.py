import numpy as np

from ..structures.branched_trajectories import BranchedTrajectories
from ..utils.numpy_utils import angle_to_vect
from .geometric_functions import (
    multiple_paths_collisions,
    path_line_collisions,
    path_semiline_collisions,
    path_semiplane_collisions,
)


def solve_boundingbox_collisions(branched_trajectories: BranchedTrajectories, bounding_box):
    """
    Compute the cut locations of a set of trajectories so that all the collisions with the bounding box
     are located after the cuts.

    Parameters
    ----------
    branched_trajectories: list of length L with (N, 2)  np.ndarray of np.float32
        Position of the trajectories

    anchors: (2, 2) np.ndarray of np.float32
        Position of the anchors of the bounding box.

    cone: (2, ) np.ndarray of np.float32
        Angular orientation of the boundaries of the cone.

    Returns
    -------

    cuts: (L, )  np.ndarray of int
        Index of the nodes where the cuts should occur on each trajectory.
    """
    cuts = np.zeros(branched_trajectories.size, dtype=int)

    for i, traj in enumerate(branched_trajectories.trajectories):
        trepassings = np.argwhere(_bounding_box_collisions(traj, bounding_box))

        if trepassings.size > 0:
            cuts[i] = trepassings[0, 0]
        else:
            cuts[i] = traj.shape - 2

    if np.any(cuts <= 1):
        cuts = np.ones(branched_trajectories.size, dtype=int)

    return cuts


def _bounding_box_collisions(trajectory, bounding_box): # TODO bouger dans geometric_functions
    cone, anchors = bounding_box.cone, bounding_box.anchors
    vect0, vect1 = angle_to_vect(cone[0]), angle_to_vect(cone[1])

    path = trajectory.emb

    collisions = np.logical_or(path_line_collisions(path, anchors[0], vect0),
                                     path_line_collisions(path, anchors[1], vect1))
    collisions[1:] = np.logical_or(collisions[1:], path_line_collisions(path[1:], anchors[0], anchors[1]-anchors[0]))

    return collisions



def solve_next_collisions(branched_trajectories, cuts, end_angles, bounds):
    embs = [traj.emb for traj in branched_trajectories.trajectories]

    cuts, still_collisions = _solve_next_boundingbox_self_overlap(embs, cuts, angle_to_vect(end_angles))
    if not still_collisions:
        cuts, still_collisions = _solve_next_boundingbox_neighbor_overlap(embs, cuts, bounds)
    if not still_collisions:
        cuts, still_collisions = _solve_next_trajectory_collisions(embs, cuts)

    return cuts, still_collisions


def _solve_next_trajectory_collisions(trajectories, cuts):
    trepassing_trajectories, trepassing_indices = multiple_paths_collisions(list(trajectories))

    still_collisions = False
    if trepassing_trajectories.size > 0:
        still_collisions = True
        collisions = np.array([np.any(trepassing_trajectories == i) for i in range(len(trajectories))])

        i = np.argmax(cuts * collisions - ~collisions)
        cuts[i] -= 1

    return cuts, still_collisions


def _solve_next_boundingbox_self_overlap(trajectories, cuts, vects):
    collisions = [np.any(path_semiplane_collisions(traj[:-1], traj[-1], vect)) for traj, vect in zip(trajectories, vects, strict=False)]

    still_collisions = False
    for i in np.argwhere(collisions):
        cuts[i] -= 1
        still_collisions = True

    return cuts, still_collisions


def _solve_next_boundingbox_neighbor_overlap(trajectories, cuts, bounds):
    still_collisions = False

    skip_next = False
    for j, bound in enumerate(bounds[1:-1]):
        vect = angle_to_vect(bound)

        ltraj, lcut = trajectories[j], cuts[j]
        rtraj, rcut = trajectories[j + 1], cuts[j + 1]

        if not skip_next:
            if np.any(path_semiline_collisions(rtraj, ltraj[-1], vect)) or np.any(
                    path_semiline_collisions(ltraj, rtraj[-1], vect)):
                still_collisions = True
                if lcut >= rcut:
                    cuts[j] -= 1
                else:
                    cuts[j + 1] -= 1
                    skip_next = True
        else:
            skip_next = False

    return cuts, still_collisions
