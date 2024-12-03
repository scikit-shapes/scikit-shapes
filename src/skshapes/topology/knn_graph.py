"""K-Nearest Neighbors graphs."""

import torch

from ..input_validation import typecheck
from ..types import Edges, int_dtype


@typecheck
def _k_ring_graph(self, k: int = 2, verbose: bool = False) -> Edges:
    """Computes the k-ring graph of the shape.

    The k-ring graph is the graph where each vertex is connected to its
    k-neighbors, where the neighborhoods are defined by the edges.

    Parameters
    ----------
    k, optional
        the size of the neighborhood, by default 2
    verbose, optional
        whether or not display information about the remaining number of steps

    Returns
    -------
        The k-ring neighbors edges.
    """
    if not k > 0:
        msg = "The number of neighbors k must be positive"
        raise ValueError(msg)

    edges = self.edges

    if verbose:
        print(f"Computing k-neighbors graph with k = {k}")  # noqa: T201
        print(f"Step 1/{k}")  # noqa: T201

    # Compute the adjacency matrix
    adjacency_matrix = torch.zeros(
        size=(self.n_points, self.n_points),
        dtype=int_dtype,
        device=self.device,
    )
    for i in range(self.n_edges):
        i0, i1 = self.edges[i]
        adjacency_matrix[i0, i1] = 1
        adjacency_matrix[i1, i0] = 1

    M = adjacency_matrix
    k_ring_edges = edges.clone()
    k_ring_edges = torch.sort(k_ring_edges, dim=1).values

    for i in range(k - 1):

        if verbose:
            print(f"Step {i+2}/{k}")  # noqa: T201

        # Iterate the adjacency matrix
        M = M @ adjacency_matrix

        # Get the connectivity (at least one path between the two vertices)
        a, b = torch.where(M > 0)

        last_edges = torch.stack((a, b), dim=1)

        # Remove the diagonal and the duplicates with the convention
        # edges[i, 0] < edges[i, 1]
        last_edges = last_edges[
            torch.where(last_edges[:, 0] - last_edges[:, 1] < 0)
        ]

        # Concatenate the new edges and remove the duplicates
        k_ring_edges = torch.cat((k_ring_edges, last_edges), dim=0)
        k_ring_edges = torch.unique(k_ring_edges, dim=0)

    return k_ring_edges


@typecheck
def _knn_graph(self, k: int = 2, include_edges: bool = False) -> Edges:
    """Returns the k-nearest neighbors edges of a shape.

    Parameters
    ----------
    shape
        The shape to process.
    k
        The number of neighbors.
    include_edges
        If True, the edges of the shape are concatenated with the k-nearest
        neighbors edges.

    Returns
    -------
    Edges
        The k-nearest neighbors edges.
    """
    if not k > 0:
        msg = "The number of neighbors k must be positive"
        raise ValueError(msg)

    from pykeops.torch import LazyTensor

    points = self.points

    points_i = LazyTensor(points[:, None, :])  # (N, 1, 3)
    points_j = LazyTensor(points[None, :, :])  # (1, N, 3)

    D_ij = ((points_i - points_j) ** 2).sum(-1)

    out = D_ij.argKmin(K=k + 1, dim=1)

    out = out[:, 1:]

    knn_edges = torch.zeros(
        (k * self.n_points, 2), dtype=int_dtype, device=self.device
    )
    knn_edges[:, 0] = torch.repeat_interleave(torch.arange(self.n_points), k)
    knn_edges[:, 1] = out.flatten()

    if include_edges and self.edges is not None:
        knn_edges = torch.cat([self.edges, knn_edges], dim=0)

    knn_edges, _ = torch.sort(knn_edges, dim=1)

    return torch.unique(knn_edges, dim=0)
