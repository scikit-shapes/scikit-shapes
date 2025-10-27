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


def compute_downstream(tree, orientation=0, reverse=False):
    pos, radius, edge_length = tree.features["pos"], tree.features["radius"], tree.features["edge_length"]
    parents = tree.parents()

    edges_vol = edge_length * (radius ** 2 + radius * radius[parents] + radius[parents] ** 2)
    vols = tree.integrate_edges(vals=edges_vol, default=2 * radius ** 3)

    edges_bar = ((radius ** 2) * pos + (radius[parents] ** 2) * pos[parents]) / (radius ** 2 + radius[parents] ** 2)
    bars = tree.integrate_edges(vals=edges_bar * edges_vol, default=pos * vols) / vols

    tree.features["downstream_volume"] = vols
    tree.features["downstream_barycenter"] = bars

    if reverse:
        tree.features["score"] = bars[:, orientation]
    else:
        tree.features["score"] = - bars[:, orientation]

    return tree


def reorder_branches(tree, scores):
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


def compute_embeddings(tree):
    if "emb" not in tree.features:
        tree.features["emb"] = np.zeros(shape=(tree.size, 2), dtype=np.float32)

    edge_lengths, angles = tree.features["edge_length"], tree.features["angle"]
    tree.features["emb"] = tree.propagate(vals=edge_lengths * np.hstack([np.cos(angles), np.sin(angles)]))

    return tree


def compute_reference_angles(tree, priority, smoothing=5, tangent_smoothing=0):
    pos = tree.features["pos"]
    parents = tree.parents()

    if tangent_smoothing > 0:
        cur_parent_level = parents
        ancesters_pos = []
        for _ in range(tangent_smoothing):
            ancesters_pos.append(pos[cur_parent_level])
            cur_parent_level = parents[cur_parent_level]
        barycenter_pos = np.array(ancesters_pos).mean(axis=0)
    else:
        barycenter_pos = pos[parents]

    tang = pos - barycenter_pos
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
        phi[tree.branches()[i,1]:tree.branches()[i,1]+4] = 0

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

    tree.features["true_angle"] = tree.propagate(phi.reshape(-1,1))

    return tree
