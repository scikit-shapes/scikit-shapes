import numpy as np
import torch
from pykeops.torch import LazyTensor
from pykeops.torch.cluster import (
    grid_cluster,
    cluster_ranges_centroids,
    sort_clusters,
    from_matrix,
)


from ..utils import diagonal_ranges
from ..types import typecheck, Points, Optional, Triangles, Number, Literal, Callable


class KeOpsSquaredDistances:
    def __init__(
        self,
        *,
        points,
        cutoff: Optional[Number] = None,
        kernel: Optional[Callable] = None,
    ):
        N = points.shape[0]
        D = points.shape[1]
        assert points.shape == (N, D)

        self.shape = (N, N)

        if cutoff is None:
            x = points
            ranges = None

        else:
            bin_size = 1.5
            # Put points into bins of size 1
            point_labels = grid_cluster(points, bin_size)

            # Compute the ranges and centroids associated to the bins
            x_ranges, x_centroids, weights_c = cluster_ranges_centroids(
                points, point_labels
            )

            # To fit the block-sparse structure of our kernel, we will need to sort
            # the points to make clusters contiguous in memory.
            sorted_labels, perm = torch.sort(point_labels.view(-1))
            sorted_points = points[perm]
            # Invert the permutation
            _, inv_perm = torch.sort(perm)
            self.perm = perm
            self.inv_perm = inv_perm

            # Compute a coarse Boolean mask:
            # print("Centroids:", x_centroids.shape)
            D2_c = ((x_centroids[:, None, :] - x_centroids[None, :, :]) ** 2).sum(2)
            cutoff_distance = np.sqrt(cutoff) + np.sqrt(D) * bin_size
            keep = D2_c < cutoff_distance**2
            print(
                f"Cutoff distance: {cutoff_distance:.2f} sigma, keep {(100.* keep).mean():.2f}% of a {keep.shape[0]:,}^2 cluster matrix"
            )
            ranges = from_matrix(x_ranges, x_ranges, keep)

            x = sorted_points

        x_i = LazyTensor(x.view(N, 1, D))
        x_j = LazyTensor(x.view(1, N, D))
        D_ij = ((x_j - x_i) ** 2).sum(-1)
        self.K_ij = kernel(D_ij)
        self.K_ij.ranges = ranges

    def __matmul__(self, other):
        assert len(other.shape) in (1, 2)
        assert other.shape[0] == self.shape[1]

        # Simple case: no cluster, etc., we can just perform the matrix product
        if not hasattr(self, "perm"):
            return self.K_ij @ other

        # Coarse-to-fine implementation that requires some shuffling
        # to fit the block-sparse structure of our kernel
        sorted_other = other[self.perm]
        assert sorted_other.shape == other.shape

        # Apply the block-sparse kernel
        conv = self.K_ij @ sorted_other

        # And don't forget to unsort the result!
        sorted_conv = conv[self.inv_perm]
        assert sorted_conv.shape == other.shape

        return sorted_conv

    @property
    def T(self):
        return self


@typecheck
def squared_distances(
    *,
    points,
    window: Literal[None, "ball", "knn", "spectral"] = None,
    cutoff: Optional[Number] = None,
    geodesic: bool = False,
    kernel: Optional[Callable] = None,
):
    """Returns the (N, N) matrix of squared distances between points."""
    # TODO: add support for batches!
    N = points.shape[0]
    D = points.shape[1]
    assert points.shape == (N, D)

    if geodesic:
        raise NotImplementedError("Geodesic distances are not implemented yet.")

    if window is None:
        return KeOpsSquaredDistances(points=points, cutoff=cutoff, kernel=kernel)

    raise NotImplementedError()
