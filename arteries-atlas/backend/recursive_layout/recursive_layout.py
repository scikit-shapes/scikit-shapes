import numpy as np

from ..structures.branched_trajectories import BranchedTrajectories
from ..structures.branching_tree import BranchingTree
from ..structures.tessellation import BoundingBox, Tessellation
from ..utils.numpy_utils import (
    integrate,
    lower_projection,
    orthogonal,
    projection,
    scalar,
)
from .geometric_functions import path_insidebox_collisions, path_semiplane_collisions
from .utils import solve_boundingbox_collisions, solve_next_collisions


def recursive_layout(tree: BranchingTree, smoothing=10):
    tree.coarse_features["cut"] = np.zeros(shape=tree.coarse_size, dtype=int)

    tessellation = Tessellation(size=tree.coarse_size, root=0, root_anchors=np.zeros((2, 2)),
                               root_cone=np.array([0, np.pi]), root_orientation=np.pi / 2)

    tree.features["angle"][tree.root] = np.pi / 2

    for junction in tree.junctions(non_terminal=True):  # Iterate over all the junctions in the tree

        children = tree.junction_children(junction)
        tessellation.partition(junction, children, proportions=tree.coarse_features["importance"][children])

        tree = embed_branches(tree, junction, tessellation, smoothing)
        extend_tessellation(tree, junction, tessellation)

    tree.features["initial_angle"] = tree.features["angle"].copy()

    # DEBUG
    anchors = np.zeros((tree.size, 2, 2))
    for junction in tree.junctions():
        node = tree.bifurcations()[junction, 0]
        anchors[node] = tessellation.boxes[junction].anchors

    return tree, anchors  # DEBUG


def embed_branches(tree, junction, tessellation, smoothing):
    """
    Embeds the branches which have the current node as their direct parent.
    The function makes sure that the embedding creates no collision with an other branch or with the bounding_box of
    the current node.

    Parameters
    ----------
    tree
    junction
    tessellation
    smoothing

    Returns
    -------

    Notes
    -------

    The algorithm avoid the collisions by determining a cut location on each branch. After the cut, the rest of the
    branch follows a straight line whose (angular) direction is defined in end_angles. Thus, the embedding step is
    decomposed as follow:

    1- Compute a first set of cuts corresponding to the bounding box collisions.
    2- While a collision between two branch embedding subsist, the longest branch involved is cut in order to remove the
    collision.
    3- Once all the collisions are solved, the resulting angle parameters, embeddings and cuts are saved in the tree
    features of the current branches.


    """
    node = tree.branch_roots(junction)

    children_boxes = tessellation.children_boxes(junction)
    branches = tree.junction_branches(junction)

    target_angles = [integrate(tree.features["angle"][node, 0], tree.features["phi"][b[1]:b[2]]) for b in branches]

    lengths = [tree.features["edge_length"][b[1]:b[2]] for b in branches]
    branched_trajectories = BranchedTrajectories(lengths, target_angles, initial_node=tree.features["emb"][node])

    limit_angles = np.array([bounding_box.orientation for bounding_box in children_boxes])  # TODO trouver plus propre

    splitting_angles = np.array([bounding_box.cone[0] for bounding_box in children_boxes])
    splitting_angles = np.append(splitting_angles, children_boxes[-1].cone[1])

    ### Step 1 ###
    cuts = solve_boundingbox_collisions(branched_trajectories, tessellation.boxes[junction])
    cut_angles(branched_trajectories, target_angles, limit_angles, cuts, smoothing)

    ### Step 2 ###
    still_collisions = True
    while still_collisions and max(cuts) > 1:
        still_collisions = False

        if not branched_trajectories.order_consistency():
            cuts.fill(1)
        else:
            cuts, still_collisions = solve_next_collisions(branched_trajectories, cuts, limit_angles, splitting_angles)

        cut_angles(branched_trajectories, target_angles, limit_angles, cuts, smoothing)

    ### Step 3 ###
    for i, (branch, child) in enumerate(zip(branches, tree.junction_children(junction), strict=False)):
        tree.features["angle"][branch[1]:branch[2]] = branched_trajectories.trajectories[i].angles
        tree.features["emb"][branch[1]:branch[2]] = branched_trajectories.trajectories[i].emb[1:]

        tree.coarse_features["cut"][child] = cuts[i]

    return tree


def cut_angles(branched_trajectories, target_angles, limit_angles, cuts, smoothing=10):
    for i, traj in enumerate(branched_trajectories.trajectories):
        cut, limit_angle = cuts[i], limit_angles[i]
        u = min(cut - 1, smoothing)

        lmbda = np.zeros(traj.angles.shape)
        lmbda[cut - 1:] = 1
        lmbda[cut - 1 - u:cut - 1, 0] = (np.arange(u) / u)

        traj.update_angles(lmbda * limit_angle + (1 - lmbda) * target_angles[i].reshape(-1, 1))


def extend_tessellation(tree: BranchingTree, junction, tessellation):  # TODO débugger
    node = tree.branch_roots(junction)
    children = tree.junction_children(junction)

    ### Set the new anchors to the branch extremities positions, and the orientation to the extremities angles
    tessellation.update(children, left_anchors=tree.features["emb"][tree.branch_ends(children - 1)],
                        right_anchors=tree.features["emb"][tree.branch_ends(children - 1)],
                        orientations=tree.features["angle"][tree.branch_ends(children - 1)])

    parent_box = tessellation.boxes[junction]
    children_boxes = tessellation.children_boxes(junction)

    ### Step 1: place the extreme anchors by projecting branch extremities on the parent bounding box ###
    place_extreme_anchor(parent_box, children_boxes[0], tree.features["emb"][:node], orient='left')
    place_extreme_anchor(parent_box, children_boxes[-1], tree.features["emb"][:node], orient='right')

    ### Step 2: place the intermediate anchors

    for j in range(len(children) - 1):
        place_intermediate_anchor(tree, children[j], children[j + 1], children_boxes[j], children_boxes[j + 1])

    ### Step 3: prune the bounding box of terminal branches, attribute their space to the other branches
    prune_terminal_boxes(tree, children, children_boxes)


def place_extreme_anchor(parent_box, child_box, previous_embeddings, orient='left'):
    idx = 0 if orient == 'left' else 1

    init_anchor = child_box.anchors[idx]

    extreme_anchor, extreme_border_vector = parent_box.anchors[idx], parent_box.border_vectors()[idx]

    child_box.anchors[idx] = projection(init_anchor, child_box.base_vector(), extreme_anchor, extreme_border_vector)

    if np.abs(child_box.orientation - parent_box.orientation) > 1e-10:
        base_projection = projection(init_anchor, child_box.base_vector(), extreme_anchor, parent_box.base_vector())

        if lower_projection(child_box.anchors[idx], base_projection, child_box.base_vector(orient=orient)):
            collisions_indices = np.argwhere(path_insidebox_collisions(previous_embeddings, child_box))

            if collisions_indices.size > 0:
                collisions_embs = previous_embeddings[collisions_indices]
                closest_collision = collisions_indices[np.argmax(scalar(collisions_embs - init_anchor,
                                                                        child_box.base_vector(orient=orient)))]
                child_box.anchors[idx] = projection(previous_embeddings[closest_collision],
                                                    extreme_border_vector,
                                                    init_anchor,
                                                    child_box.base_vector(orient=orient))


def place_intermediate_anchor(tree, left_child: int, right_child: int, left_box: BoundingBox, right_box: BoundingBox):
    left_imp = np.log(tree.coarse_features["importance"][left_child] + 2)
    right_imp = np.log(tree.coarse_features["importance"][right_child] + 2)

    middle_anchor = (right_imp * left_box.anchors[1] + left_imp * right_box.anchors[0]) / (left_imp + right_imp)
    left_branch, right_branch = tree.branches(left_child - 1), tree.branches(right_child - 1)

    left_emb = tree.features["emb"][left_branch[1]:left_branch[2]]  # TODO vérifier
    right_emb = tree.features["emb"][right_branch[1]:right_branch[2]]

    border_vector = left_box.border_vectors()[1]
    border_normal_vector = orthogonal(border_vector)

    # Correct eventual collisions
    collision_indices = np.argwhere(
        path_semiplane_collisions(right_emb, left_box.anchors[1], left_box.orientation_vector()))

    if collision_indices.size > 0:
        i = np.argmin(scalar(right_emb[collision_indices[:, 0]], border_normal_vector))
        leftest = right_emb[collision_indices[i, 0]]

        if lower_projection(leftest, middle_anchor, border_normal_vector):
            middle_anchor = projection(left_box.anchors[1], left_box.base_vector(), leftest,
                                       left_box.border_vectors()[1])

    else:  # TODO vérifier qu'on ne peut pas être dans les deux cas à la fois
        collision_indices = np.argwhere(
            path_semiplane_collisions(left_emb, right_box.anchors[0], right_box.orientation_vector()))

        if collision_indices.size > 0:
            i = np.argmin(scalar(left_emb[collision_indices[:, 0]], - border_normal_vector))
            rightest = left_emb[collision_indices[i, 0]]

            if lower_projection(rightest, middle_anchor, - border_normal_vector):
                middle_anchor = projection(right_box.anchors[0], right_box.base_vector(), rightest,
                                           right_box.border_vectors()[0])

    left_box.anchors[1] = projection(left_box.anchors[1], left_box.base_vector(), middle_anchor, border_vector)
    right_box.anchors[0] = projection(right_box.anchors[0], right_box.base_vector(), middle_anchor, border_vector)


def prune_terminal_boxes(tree, children, children_boxes):
    terminals = tree.out_degrees()[tree.branch_ends(children - 1)] == 0
    already_pruned = np.zeros(len(children), dtype=bool)

    scores = np.log(tree.coarse_features["importance"][children] + 2)[:, 0]
    for i in np.argsort(-scores):
        if not terminals[i]:
            current_box = children_boxes[i]
            current_anchors = current_box.anchors.copy()

            left = i - 1
            while left >= 0 and terminals[left] and not already_pruned[left]:
                if lower_projection(children_boxes[left].anchors[1], current_anchors[0],
                                    children_boxes[left].border_vectors()[1]):

                    current_anchors[0] = projection(current_anchors[0], current_box.base_vector(),
                                                    children_boxes[left].anchors[0], children_boxes[left].border_vectors()[0])

                    current_box.update(left_cone=children_boxes[left].cone[0])
                    children_boxes[left].update(right_anchor=children_boxes[left].anchors[0],
                                             right_cone=children_boxes[left].cone[0])

                    already_pruned[left] = True
                else:
                    break
                left -= 1

            right = i + 1
            while right <= len(children) - 1 and terminals[right] and not already_pruned[right]:
                if lower_projection(children_boxes[right].anchors[0], current_anchors[1],
                                    children_boxes[right].border_vectors()[0]):
                    current_anchors[1] = projection(current_anchors[1], current_box.base_vector(),
                                                    children_boxes[right].anchors[1], children_boxes[right].border_vectors()[1])

                    current_box.update(right_cone=children_boxes[right].cone[1])
                    children_boxes[right].update(left_anchor=children_boxes[right].anchors[1],
                                             left_cone=children_boxes[right].cone[1])

                    already_pruned[right] = True
                else:
                    break
                right += 1

            current_box.update(left_anchor=current_anchors[0], right_anchor=current_anchors[1])
