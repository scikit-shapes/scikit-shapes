# ruff: noqa: EXE002

import numpy as np
import torch
from pykeops.torch import LazyTensor

from ..tree_creation import compute_embeddings


def structure_force(tree):
    force = (tree.features["initial_angle"] - tree.features["angle"])
    force[tree.root] = 0

    return force


def bending_force(tree, mu=2):
    angles, phi, edge_lengths, radius = tree.features["angle"], tree.features["phi"], tree.features["edge_length"], \
    tree.features["radius"]
    parents = tree.parents()

    force = (radius ** mu) * (phi + angles[parents] - angles) / (edge_lengths + edge_lengths[parents] + 1e-6)
    force[tree.out_degrees() > 1] = 0

    return force


def repulsion_force(tree, sigma=300):
    embs, angles, edge_lengths = tree.features["emb"], tree.features["angle"], tree.features["edge_length"]

    embs_i = LazyTensor(torch.tensor(embs[:, None, :], dtype=torch.float32))
    embs_j = LazyTensor(torch.tensor(embs[None, :, :], dtype=torch.float32))

    deltas = embs_j - embs_i
    dists = deltas.norm(dim=-1)

    K_ij = (-(dists ** 2) / sigma).exp()
    reduction = (LazyTensor.cat((K_ij * deltas * (0.1 / (dists + 1e-6) + 1 / (dists ** 2 + 1e-6)), K_ij), dim=-1)).sum(
        axis=1)

    forces = - reduction[:, :2] / (reduction[:, 2:] - 0.999)
    forces = forces.cpu().numpy()

    forces = tree.integrate_nodes(vals=forces)
    forces = forces / tree.descendant_count().astype(np.float32)

    angular_forces = (forces * np.hstack([-np.sin(angles), np.cos(angles)])).sum(axis=1, keepdims=True)
    angular_forces[tree.root] = 0
    return angular_forces / (edge_lengths + 1e-6)


def compute_force(tree, iters=200, alpha=0.1, beta=0.1, gamma=0.1, mu=2, sigma=300, momentum=1, clip=None):
    old_angles = tree.features["angle"].copy()

    for i in range(iters):
        if momentum > 0:
            b = momentum * i / (i + 3)
            cur_angles = tree.features["angle"].copy()
            tree.features["angle"] = tree.features["angle"] + b * (tree.features["angle"] - old_angles)
            old_angles = cur_angles

        total_force = alpha * repulsion_force(tree, sigma) + beta * structure_force(tree) + gamma * bending_force(tree,
                                                                                                                   mu)

        if clip is not None:
            total_force = total_force.clip(-clip, clip)

        tree.features["angle"] += total_force

        for node in tree.bifurcations(non_terminal=True)[:, 0]:
            children = tree.node_children(node)

            for branch in tree.branches()[tree.branch_index()[children]]:
                begin_junction = branch[1]
                end_junction = branch[1] + max(tree.coarse_features["cut"][tree.bifurcations_index(branch[1])] - 15, 0)

                u = min(end_junction - begin_junction, 30)
                t = 0.1 + np.zeros((end_junction - begin_junction, 1))
                t[-u:, 0] = np.minimum(1, 0.1 + (np.arange(u) / u))

                tree.features["angle"][begin_junction:end_junction] = t * tree.features["angle"][begin_junction:end_junction] + (1 - t) * (
                            tree.features["angle"][branch[0]] + tree.features["true_angle"][begin_junction:end_junction])

        compute_embeddings(tree)

    return tree
