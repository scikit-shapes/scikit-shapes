# ruff: noqa: EXE002
# ruff: noqa: I001

import numpy as np


class Tree:
    def __init__(self, adj_matrix, features: dict, root: int):
        self.adjmatrix = adj_matrix
        self.features = features
        self.root = root

        self.size = adj_matrix.shape[0]

        self._parents = None
        self._children = None

        self._out_degrees = None
        self._bifurcations = None
        self._bifurcations_nonterminal = None
        self._bifurcationsindex = None

        self._leaves = None

    def reorder(self, order):
        self.adjmatrix = self.adjmatrix[order, :][:, order]
        self.features = {k: self.features[k].copy()[order] for k in self.features}

        for i, j in zip(self.adjmatrix.indptr[:-1], self.adjmatrix.indptr[1:], strict=False):
            self.adjmatrix.indices[i:j] = np.sort(self.adjmatrix.indices[i:j])

        self._parents, self._out_degrees =  None, None
        self._bifurcations, self._bifurcations_nonterminal, self._leaves = None, None, None


    def parents(self):
        if self._parents is None:
            c = self.adjmatrix.tocoo()

            self._parents = np.zeros(shape=self.size, dtype=int)
            self._parents[c.col] = c.row

        return self._parents

    def node_children(self, node):
        return self.adjmatrix.indices[slice(self.adjmatrix.indptr[node], self.adjmatrix.indptr[node + 1])]

    def out_degrees(self):
        if self._out_degrees is None:
            self._out_degrees = np.array(self.adjmatrix.sum(axis=1))

        return self._out_degrees

    def bifurcations(self, non_terminal=False):
        if non_terminal:
            if self._bifurcations_nonterminal is None:
                self._bifurcations_nonterminal = np.argwhere(self.out_degrees()[:, 0] > 1)

                if self.root not in self._bifurcations_nonterminal:
                    self._bifurcations_nonterminal = np.concatenate([[[self.root]], self._bifurcations_nonterminal])

            return self._bifurcations_nonterminal
        else:
            if self._bifurcations is None:
                self._bifurcations = np.argwhere(self.out_degrees()[:, 0] != 1)

                if self.root not in self._bifurcations:
                    self._bifurcations = np.concatenate([[[self.root]], self._bifurcations])

            return self._bifurcations

    def bifurcations_index(self, node: None | int | list = None):
        if self._bifurcationsindex is None:
            self._bifurcationsindex = np.zeros(self.size, dtype=int)

            for i, n in enumerate(self.bifurcations()):
                self._bifurcationsindex[n] = i

        return self._bifurcationsindex if node is None else self._bifurcationsindex[node]

    def leaves(self):
        if self._leaves is None:
            self._leaves = np.argwhere(self.out_degrees()[:, 0] == 0)

        return self._leaves
