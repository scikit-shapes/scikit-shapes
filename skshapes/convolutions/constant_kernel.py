import torch
from ..types import typecheck, Points, Optional, Triangles, Number, Literal


class Constant1Kernel:
    """A summation operator that stands for torch.ones(N, N)."""

    def __init__(self, points, target_points=None):
        if target_points is None:
            target_points = points

        N = points.shape[0]
        M = target_points.shape[0]
        D = points.shape[1]
        assert points.shape == (N, D)
        assert target_points.shape == (M, D)

        self.shape = (N, M)

    def __matmul__(self, other):
        assert len(other.shape) in (1, 2)
        assert other.shape[0] == self.shape[1]
        sums = other.sum(dim=0, keepdim=True)
        if len(other.shape) == 1:
            assert sums.shape == (1,)
            sums = sums.tile(self.shape[0])
            assert sums.shape == (self.shape[0],)
        elif len(other.shape) == 2:
            assert sums.shape == (1, other.shape[1])
            sums = sums.tile(self.shape[0], 1)
            assert sums.shape == (self.shape[0], other.shape[1])
        return sums

    @property
    def T(self):
        return self


@typecheck
def constant_1_kernel(*, points, target_points=None, **kwargs) -> Constant1Kernel:
    """Returns the (N, N) matrix of squared distances between points.

    For geodesic kernels, we may want to stick to connected components?
    """
    return Constant1Kernel(points, target_points)
