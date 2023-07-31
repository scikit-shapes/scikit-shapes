import torch
from pykeops.torch import LazyTensor

from ..utils import diagonal_ranges
from ..types import typecheck, Points, Optional, Triangles, Number, Literal


class KeOpsSquaredDistances:
    pass


@typecheck
def squared_distances(
    *,
    points,
    window: Literal[None, "ball", "knn", "spectral"] = None,
    cutoff: Optional[Number] = None,
    geodesic: bool = False,
):
    """Returns the (N, N) matrix of squared distances between points."""
    # TODO: add support for batches!
    N = points.shape[0]
    D = points.shape[1]
    assert points.shape == (N, D)

    if geodesic:
        raise NotImplementedError("Geodesic distances are not implemented yet.")

    if window is None:
        x_i = LazyTensor(points.view(N, 1, D))
        x_j = LazyTensor(points.view(1, N, D))
        D_ij = ((x_j - x_i) ** 2).sum(-1)
        return D_ij

    raise NotImplementedError()
