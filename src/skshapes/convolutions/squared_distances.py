"""Squared distances between points."""

from collections.abc import Callable
from typing import Literal

import numpy as np
import torch
from pykeops.torch import LazyTensor
from pykeops.torch.cluster import (
    cluster_ranges_centroids,
    from_matrix,
    grid_cluster,
)

from ..input_validation import typecheck
from ..types import (
    Number,
)


class KeOpsSquaredDistances:
    """Squared distances between points using KeOps.

    Parameters
    ----------
    points
        The points in the source space.
    cutoff
        The cutoff value for the window.
    kernel
        The kernel to use.
    target_points
        The points in the target space.

    """

    def __init__(
        self,
        *,
        points,
        cutoff: Number | None = None,
        kernel: Callable | None = None,
        target_points=None,
    ):
        if target_points is None:
            target_points = points

        M = target_points.shape[0]
        N = points.shape[0]
        D = points.shape[1]
        assert points.shape == (N, D)
        assert target_points.shape == (M, D)

        self.shape = (M, N)

        if cutoff is None:
            x = points
            y = target_points
            ranges = None

        else:
            if M != N or not torch.allclose(points, target_points):
                raise NotImplementedError(
                    "Cutoff argument is not available yet when computing"
                    + "convolutions between different point clouds."
                )

            bin_size = 1.5
            # Put points into bins of size 1
            point_labels = grid_cluster(points, bin_size)

            # Compute the ranges and centroids associated to the bins
            x_ranges, x_centroids, weights_c = cluster_ranges_centroids(
                points, point_labels
            )

            # To fit the block-sparse structure of our kernel, we will need to
            # sort the points to make clusters contiguous in memory.
            sorted_labels, perm = torch.sort(point_labels.view(-1))
            sorted_points = points[perm]
            # Invert the permutation
            _, inv_perm = torch.sort(perm)
            self.perm = perm
            self.inv_perm = inv_perm

            # Compute a coarse Boolean mask:
            # print("Centroids:", x_centroids.shape)
            D2_c = (
                (x_centroids[:, None, :] - x_centroids[None, :, :]) ** 2
            ).sum(2)
            cutoff_distance = np.sqrt(cutoff) + np.sqrt(D) * bin_size
            keep = D2_c < cutoff_distance**2
            ranges = from_matrix(x_ranges, x_ranges, keep)

            x = sorted_points
            y = x

        x_i = LazyTensor(x.view(1, N, D))
        y_j = LazyTensor(y.view(M, 1, D))
        D_ij = ((y_j - x_i) ** 2).sum(-1)
        self.K_ij = kernel(D_ij)
        self.K_ij.ranges = ranges

    def sum(self, *args, **kwargs):
        """Sum of the kernel."""
        return self.K_ij.sum(*args, **kwargs)

    def __matmul__(self, other):
        """Matrix multiplication with a vector or matrix.

        Parameters
        ----------
        other
            The vector or matrix to multiply with.

        Returns
        -------
        torch.Tensor
            The result of the matrix multiplication.
        """
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
        """Transpose of the kernel."""
        return self


@typecheck
def squared_distances(
    *,
    points,
    window: Literal[None, "ball", "knn", "spectral"] = None,
    cutoff: Number | None = None,
    geodesic: bool = False,
    kernel: Callable | None = None,
    target_points=None,
):
    """Matrix of squared distances between points.

    If source_points is not None, then the (N, M) matrix of squared distances
    between points (in rows) and source_points (in columns) is returned.

    Else, the (N, N) matrix of squared distances between points is returned.

    Parameters
    ----------
    points
        The points in the target space.
    window
        The type of window to use.
    cutoff
        The cutoff value for the window.
    geodesic
        Whether to use geodesic distances.
    kernel
        The kernel to use.
    target_points
        The points in the source space. If None, then the points in the target
        space are used.

    """
    if target_points is None:
        target_points = points

    # TODO: add support for batches!
    N = points.shape[0]
    M = target_points.shape[0]
    D = points.shape[1]
    assert points.shape == (N, D)
    assert target_points.shape == (M, D)

    if geodesic:
        msg = "Geodesic distances are not implemented yet."
        raise NotImplementedError(msg)

    if window is None:
        return KeOpsSquaredDistances(
            points=points,
            cutoff=cutoff,
            kernel=kernel,
            target_points=target_points,
        )

    raise NotImplementedError()
