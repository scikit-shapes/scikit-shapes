import torch
from pykeops.torch import LazyTensor

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
            x_i = LazyTensor(points.view(N, 1, D))
            x_j = LazyTensor(points.view(1, N, D))
            D_ij = ((x_j - x_i) ** 2).sum(-1)
            self.K_ij = kernel(D_ij)

    def __matmul__(self, other):
        assert len(other.shape) in (1, 2)
        assert other.shape[0] == self.shape[1]
        return self.K_ij @ other

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
