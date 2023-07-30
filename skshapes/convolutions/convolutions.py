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
    def T(self):
        return LinearOperator(
            self.matrix.T,
            input_scaling=self.output_scaling,
            output_scaling=self.input_scaling,
        )

    @property
    def shape(self):
        return self.matrix.shape


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
        raise NotImplementedError("Geodesic distances not implemented yet.")

    if window is None:
        x_i = LazyTensor(points.view(N, 1, D))
        x_j = LazyTensor(points.view(1, N, D))
        D_ij = ((x_j - x_i) ** 2).sum(-1)
        return D_ij

    raise NotImplementedError()


@typecheck
def _point_convolution(
    self,
    *,
    kernel: Literal["uniform", "gaussian"] = "gaussian",
    scale: Optional[Number] = None,
    window: Literal[None, "ball", "knn", "spectral"] = None,
    cutoff: Optional[Number] = None,
    geodesic: bool = False,
    normalize: bool = False,
    dtype: Optional[Literal["float", "double"]] = None,
) -> LinearOperator:
    """Creates a convolution kernel on a PolyData as a (N, N) linear operator."""
    N = self.n_points
    weights_j = self.point_weights
    assert weights_j.shape == (N,)

    X = self.points

    if dtype == "float":
        X = X.float()
        weights_j = weights_j.float()
    elif dtype == "double":
        X = X.double()
        weights_j = weights_j.double()

    # Divisions are expensive: whenever possible, it's best to scale the points
    # ahead of time instead of scaling the distances for every pair of points.
    if scale is not None:
        if kernel == "gaussian":
            sqrt_2 = 1.41421356237
            X = X / (sqrt_2 * scale)
        elif kernel == "uniform":
            X = X / scale

    D_ij = squared_distances(points=X, window=window, cutoff=cutoff, geodesic=geodesic)

    if scale is None:
        # scale = +infinity, the kernel is always equal to 1
        K_ij = 1.0 * (D_ij + 1).step()

    else:
        if kernel == "gaussian":
            K_ij = (-D_ij).exp()
        elif kernel == "uniform":
            K_ij = 1.0 * (D_ij <= 1)

    if normalize:
        total_weights_i = K_ij @ weights_j
        assert total_weights_i.shape == (N,)
        norm_i = 1.0 / total_weights_i

    else:
        norm_i = None

    return LinearOperator(K_ij, input_scaling=weights_j, output_scaling=norm_i)
