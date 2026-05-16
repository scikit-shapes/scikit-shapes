# ruff: noqa: EXE002

import numpy as np
import scipy

from .tree import Tree


class BranchingTree(Tree):
    def __init__(self, adjmatrix, features, root, pruning=0, internal_pruning=0):
        self.pruning, self.internal_pruning = pruning, internal_pruning

        self._branches = None
        self._branches_r = None
        self._branchindex = None
        self._descendants = None

        # Reorder nodes by dfs order
        dfs_order = scipy.sparse.csgraph.depth_first_order(adjmatrix, root, directed=True, return_predecessors=False)

        ordered_adjmatrix = adjmatrix[dfs_order, :][:, dfs_order]
        ordered_features = {k: features[k].copy()[dfs_order] for k in features}

        Tree.__init__(self, ordered_adjmatrix, ordered_features, 0)

        self.prune(pruning, internal_pruning)
        self.coarse_features = {}

        self.coarse_size = self.bifurcations().shape[0]

    def prune(self, pruning=0, internal_pruning=0):  # TODO vérifier tout ça
        branches = self.branches()
        branch_degrees = self.out_degrees().squeeze()[branches[:, 2] - 1]

        pruned_leaves = np.logical_and(branch_degrees == 0, branches[:, 2] - branches[:, 1] <= pruning)
        pruned_internal = np.logical_and(branch_degrees >= 1, branches[:, 2] - branches[:, 1] <= internal_pruning)

        self.adjmatrix = self.adjmatrix.tolil()

        keep = np.ones(self.size, dtype=bool)
        for i in range(branches.shape[0] - 1, -1, -1):
            if pruned_leaves[i]:
                keep[branches[i, 1]:branches[i, 2]] = False
            elif pruned_internal[i]:
                keep[branches[i, 1]:branches[i, 2]] = False
                self.adjmatrix[branches[i, 0]] += self.adjmatrix[branches[i, 2] - 1]

        self.adjmatrix = self.adjmatrix.tocsr()[keep, :][:, keep]
        self.features = {k: self.features[k][keep] for k in self.features}
        self.size = self.adjmatrix.shape[0]

        self._bifurcations, self._bifurcations_nonterminal, self._out_degrees = None, None, None
        self._parents, self._branches, self._branches_r = None, None, None

    def reorder(self, order: list):
        super().reorder(order)
        self._branches, self._branches_r, self._branchindex, self._descendants = None, None, None, None

        if self.coarse_features:  # TODO vérifier
            self.coarse_features = {k: self.coarse_features[k][order[self.branch_roots()]] for k in self.coarse_features}

    def branches(self, index: None | int | list = None, reverse=False):
        if self._branches is None or self._branches_r is None:
            parents, bifurcations = self.parents(), self.bifurcations()

            # (Branch Parent, Branch start, Branch end)
            self._branches = np.stack([parents[bifurcations[:-1] + 1], bifurcations[:-1] + 1, bifurcations[1:] + 1],
                                      axis=1).squeeze()

            self._branches_r = np.flip(self._branches.copy(), axis=0)
            self._branches_r[:, 1:] -= 1

        if reverse:
            return self._branches_r if index is None else self._branches_r[index, :]
        else:
            return self._branches if index is None else self._branches[index, :]

    def branch_index(self, node: None | int | list = None):  # TODO implémenter reverse ?
        if self._branchindex is None:
            self._branchindex = np.zeros(self.size, dtype=int)

            for i, branch in enumerate(self.branches()):
                self._branchindex[branch[1]:branch[2]] = i

        return self._branchindex if node is None else self._branchindex[node]

    def branch_roots(self, branch: None | int | list = None):
        return self.branches()[:, 0] if branch is None else self.branches()[branch, 0]

    def branch_ends(self, branch: None | int | list = None):
        return self.branches()[:, 2] - 1 if branch is None else self.branches()[branch, 2] - 1


    def junctions(self, non_terminal=False): # TODO corriger ça
        bifurcations = self.bifurcations(non_terminal)
        return self.bifurcations_index(bifurcations).reshape(-1)

    def junction_children(self, junction):
        node = self.bifurcations()[junction, 0] # TODO vérifier que tout va bien
        node_children = self.node_children(node)
        branch_extreme_bifurcations = self.branch_ends(self.branch_index(node_children))

        return self.bifurcations_index(branch_extreme_bifurcations)


    def junction_branches(self, junction):
        node = self.branch_roots(junction) # TODO debug
        branch_indices = self.branch_index(self.node_children(node))
        return self.branches(branch_indices)
        # return self.branches(self.node_children(node))

    def descendant_count(self):  # TODO harmoniser dimensions
        if self._descendants is None:
            descendants = np.ones(shape=self.size, dtype=int)

            for branch in self.branches(reverse=True):
                descendants[branch[2]:branch[1]:-1] = np.arange(descendants[branch[2]],
                                                                descendants[branch[2]] + branch[2] - branch[1])
                descendants[branch[0]] += descendants[branch[1] + 1]

            self._descendants = np.expand_dims(descendants, axis=1)

        return self._descendants

    def propagate(self, vals):
        out = np.zeros(vals.shape, dtype=np.float32)
        out[self.root] = vals[self.root]

        for branch in self.branches():
            # out[branch[1]:branch[2], :] = out[branch[0]] + np.cumsum(vals[branch[1]:branch[2]], axis=0)
            vals[branch[1]] += out[branch[0], :]
            out[branch[1]:branch[2]] = np.cumsum(vals[branch[1]:branch[2]], axis=0)

        return out

    def integrate_nodes(self, vals):
        out = vals.copy()

        for branch in self.branches(reverse=True):
            out[branch[2]:branch[1]:-1] = np.cumsum(out[branch[2]:branch[1]:-1], axis=0)
            out[branch[0]] += out[branch[1] + 1]

        return out

    def integrate_edges(self, vals, default=None):
        out = np.zeros(vals.shape, dtype=np.float32)
        if default is not None:
            out[self.leaves()] = default[self.leaves()]

        for branch in self.branches(reverse=True):
            out[branch[2] - 1:branch[1]:-1] = out[branch[2]] + np.cumsum(vals[branch[2]:branch[1] + 1:-1], axis=0)
            out[branch[0]] += out[branch[1] + 1] + vals[branch[1] + 1]

        return out
