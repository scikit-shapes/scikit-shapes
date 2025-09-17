import numpy as np

from .structures.branching_tree import BranchingTree
from .utils.numpy_utils import sort


def initialize_vascular_tree(adjmatrix, features, root, pruning=0, internal_pruning=0):
    tree = BranchingTree(adjmatrix, features, root, pruning=pruning, internal_pruning=internal_pruning)
    tree.features["radius"][tree.out_degrees() == 0] = 1e-5
    return tree


def compute_edge_lenghts(tree):
    pos = tree.features["pos"]

    tree.features["edge_length"] = np.linalg.norm(pos[tree.parents()] - pos, axis=-1, keepdims=True)
    tree.features["edge_length"][tree.root] = 0

    return tree


def compute_downstream(tree):
    pos, radius, edge_length = tree.features["pos"], tree.features["radius"], tree.features["edge_length"]
    parents = tree.parents()

    edges_vol = edge_length * (radius ** 2 + radius * radius[parents] + radius[parents] ** 2)
    vols = tree.integrate_edges(vals=edges_vol, default=2 * radius ** 3)

    edges_bar = ((radius ** 2) * pos + (radius[parents] ** 2) * pos[parents]) / (radius ** 2 + radius[parents] ** 2)
    bars = tree.integrate_edges(vals=edges_bar * edges_vol, default=pos * vols) / vols

    tree.features["downstream_volume"] = vols
    tree.features["downstream_barycenter"] = bars
    tree.features["score"] = - bars[:, 0]

    return tree


def reorder_branches(tree, scores): # TODO vérifier

    order = np.arange(tree.size)
    sectors = np.arange(tree.size) + 1

    for node in np.flip(tree.bifurcations(non_terminal=True)[:, 0]):
        children = sort(tree.node_children(node), scores)
        branches = tree.branches()[tree.branch_index()[children]]

        start, end = np.min(children), np.max(sectors[branches[:, 2] - 1])

        order[start:end] = np.concatenate([order[child:sectors[branches[i, 2]-1]] for i, child in enumerate(children)])
        sectors[node] = end

    tree.reorder(order)

    return tree


def initialize_angles(tree):
    # TODO nettoyer à terme
    scores, vols = tree.features["score"], tree.features["downstream_volume"]

    rel_angles = np.zeros((tree.size, 2), dtype=np.float32)
    rel_angles[tree.root] = [0, 1]

    for node in tree.bifurcations(non_terminal=True)[:, 0]:
        children = tree.node_children(node)
        children = children[np.argsort(scores[children])]

        cum_angles = np.cumsum(vols[children])
        cum_angles = np.concatenate([[0], cum_angles / cum_angles[-1]])

        rel_angles[children, 0], rel_angles[children, 1] = cum_angles[:-1], cum_angles[1:]

    angle_ranges = np.zeros(shape=(tree.size, 2), dtype=np.float32)
    angle_ranges[tree.root] = [0, np.pi]

    for branch in tree.branches():
        branch_range = angle_ranges[branch[0], 0] + (angle_ranges[branch[0], 1] - angle_ranges[branch[0], 0]) * \
                       rel_angles[branch[1]]
        angle_ranges[branch[1]:branch[2]] = branch_range

    angles = np.expand_dims((angle_ranges[:, 0] + angle_ranges[:, 1]) / 2, axis=-1)

    tree.features["range"] = (angle_ranges[:, 1] - angle_ranges[:, 0]) / np.maximum(1, tree.out_degrees()).squeeze() # DEBUG
    tree.features["angle"] = angles
    tree.features["initial_angle"] = tree.features["angle"].copy()

    return tree


def compute_embeddings(tree):
    if "emb" not in tree.features:
        tree.features["emb"] = np.zeros(shape=(tree.size, 2), dtype=np.float32)

    edge_lengths, angles = tree.features["edge_length"], tree.features["angle"]
    tree.features["emb"] = tree.propagate(vals=edge_lengths * np.hstack([np.cos(angles), np.sin(angles)]))

    return tree


def compute_reference_angles(tree, priority, smoothing=5):
    pos = tree.features["pos"]
    parents = tree.parents()

    tang = pos - pos[parents]
    tang = tang / (np.linalg.norm(tang, axis=1, keepdims=True) + 1e-6)

    normals = np.zeros(tang.shape)
    tang[tree.root], normals[tree.root] = [0, 0, 1], [1, 0, 0]

    for i in range(1, tree.size):
        normals[i] = normals[parents[i]] - np.dot(normals[parents[i]], tang[i]) * tang[i]
        normals[i] = normals[i] / (np.linalg.norm(normals[i]) + 1e-6)

    phi = np.arccos((tang[tree.parents()] * tang).sum(axis=-1).clip(0, 1))
    phi[(normals[tree.parents()] * tang).sum(axis=-1) < 0] = - phi[(normals[tree.parents()] * tang).sum(axis=-1) < 0]

    ## Correct root angle
    for i in  tree.branch_index()[tree.node_children(tree.root)]:
        phi[tree.branches()[i,1]:tree.branches()[i,1]+4] = 0 # TODO débugger

    if smoothing > 0:
        for branch in tree.branches():
            window = min(smoothing, branch[2] - branch[1])
            ker = np.ones(window)

            weight = np.convolve(np.ones(branch[2] - branch[1]), ker, "same")
            phi[branch[1]:branch[2]] = np.convolve(phi[branch[1]:branch[2]], ker, "same") / weight



    # Change the curvature values at the junction to make them compatible with the new children order
    for node in tree.bifurcations(non_terminal=True)[:, 0]:
        if tree.out_degrees()[node] > 1:
            children = tree.node_children(node)
            branches = tree.branches()[tree.branch_index()[children]]

            i0, i1 = np.argsort(-priority[children], axis=0)[[0, 1], 0]
            if (i1 - i0) * (phi[children[i1]] - phi[children[i0]]) < 0:
                for branch in branches:
                    phi[branch[1]:branch[2]] = - phi[branch[1]:branch[2]]

            idx = np.arange(len(children))
            if np.any(np.argsort(phi[children]) != idx):
                phi[children] =  ((idx - i0) * phi[children[i1]] + (i1 - idx) * phi[children[i0]]) / (i1 - i0)

        tree.features["phi"] = phi.copy().reshape(-1, 1)

    tree.features["true_angle"] = tree.propagate(phi.reshape(-1,1)) # TODO debug

    return tree
