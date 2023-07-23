import torch
from pykeops.torch import LazyTensor

from ..utils import diagonal_ranges
from ..types import typecheck, Points, Optional, Triangles, Number, Literal


class LinearOperator:
    """A simple wrapper for scaled linear operators."""

    def __init__(self, matrix, input_scaling=None, output_scaling=None):
        N, M = matrix.shape
        assert matrix.shape == (N, M)
        assert input_scaling is None or input_scaling.shape == (N,)
        assert output_scaling is None or output_scaling.shape == (M,)

        self.matrix = matrix
        self.input_scaling = input_scaling
        self.output_scaling = output_scaling

    def __matmul__(self, other):
        assert len(other.shape) in (1, 2)
        assert other.shape[0] == self.matrix.shape[1]

        i_s = self.input_scaling if self.input_scaling is not None else 1
        o_s = self.output_scaling if self.output_scaling is not None else 1

        if len(other.shape) == 2:
            i_s = i_s.view(-1, 1) if i_s is not None else 1
            o_s = o_s.view(-1, 1) if o_s is not None else 1

        return o_s * (self.matrix @ (i_s * other))

    @property
    def shape(self):
        return self.matrix.shape


@typecheck
def squared_distances(
    *,
    points: Points,
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
        raise NotImplementedError("Geodesic distances not implemented yet.")

    if window is None:
        x_i = LazyTensor(points.view(N, 1, D))
        x_j = LazyTensor(points.view(1, N, D))
        D_ij = ((x_j - x_i) ** 2).sum(-1)
        return D_ij

    raise NotImplementedError()


@typecheck
def point_convolution(
    self,
    *,
    kernel: Literal["gaussian", "uniform"] = "gaussian",
    scale: Number = 1.0,
    window: Literal[None, "ball", "knn", "spectral"] = None,
    cutoff: Optional[Number] = None,
    geodesic: bool = False,
    normalize: bool = False,
) -> LinearOperator:
    """Creates a convolution kernel on a PolyData as a (N, N) linear operator."""
    N = self.n_points
    weights_j = self.point_weights
    assert weights_j.shape == (N,)

    D_ij = squared_distances(
        points=self.points, window=window, cutoff=cutoff, geodesic=geodesic
    )

    if kernel == "gaussian":
        K_ij = (-D_ij / (2 * scale**2)).exp()
    elif kernel == "uniform":
        K_ij = 1.0 * (D_ij <= scale**2)

    if normalize:
        total_weights_i = K_ij @ weights_j
        assert total_weights_i.shape == (N,)
        norm_i = 1.0 / total_weights_i

    else:
        norm_i = None

    return LinearOperator(K_ij, input_scaling=weights_j, output_scaling=norm_i)
