import torch
from pykeops.torch import LazyTensor

from ..utils import diagonal_ranges
from ..types import typecheck, Points, Optional, Triangles, Number


def structure_tensors(*, points: Points, scale=1.0, ranges=None):
    """Compute the structure tensors of a point cloud.

    Args:
        points (Tensor): (N, 3) coordinates of the points or mesh vertices.
        scale (float, optional): smoothing scale. Defaults to 1.

    Returns:
        (Tensor): (N, 3, 3) tensor of structure tensors.
    """
    # Number of points:
    N = points.shape[0]
    points = points / scale

    # Encode as symbolic tensors:
    # Points:
    x_i = LazyTensor(points.view(N, 1, 3))
    x_j = LazyTensor(points.view(1, N, 3))

    # Squared distance:
    d2_ij = ((x_j - x_i) ** 2).sum(-1)  # (N, N, 1)
    # Gaussian window:
    window_ij = (-d2_ij / 2).exp()  # (N, N, 1)

    # Compute the local average and total weight in the neighborhood:
    xmean = (window_ij * x_j).sum(dim=1)
    wsum = window_ij.sum(dim=1)
    xmean = xmean / wsum

    assert xmean.shape == (N, 3)
    assert wsum.shape == (N, 1)

    xmean_i = LazyTensor(xmean.view(N, 1, 3))

    # Compute the structure tensors:
    # (N, N, 3, 3)
    ST_ij = window_ij * (x_j - xmean_i).tensorprod(x_j - xmean_i)
    # Reduction:
    if ranges is not None:
        ST_ij.ranges = ranges

    ST = ST_ij.sum(1)  # (N, 3, 3)
    ST = ST.view(N, 3, 3) / wsum.view(N, 1, 1)
    return ST
