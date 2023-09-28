import numpy as np
import torch

from ..types import typecheck, Points, Optional, Triangles, Number, Literal
from .squared_distances import squared_distances
from .constant_kernel import constant_1_kernel


class LinearOperator:
    """A simple wrapper for scaled linear operators."""

    def __init__(self, matrix, input_scaling=None, output_scaling=None):
        M, N = matrix.shape
        assert matrix.shape == (M, N)
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
    target=None,
) -> LinearOperator:
    """Creates a convolution kernel on a PolyData as a (N, N) linear operator if no target is provided,
    or as a (M, N) linear operator if a target is provided"""

    if target is None:
        target = self

    N = self.n_points
    M = target.n_points

    weights_j = self.point_weights
    assert weights_j.shape == (N,)

    X = self.points
    Y = target.points

    if dtype == "float":
        X = X.float()
        Y = Y.float()
        weights_j = weights_j.float()
    elif dtype == "double":
        X = X.double()
        Y = Y.double()
        weights_j = weights_j.double()

    backend_args = dict(window=window, cutoff=cutoff, geodesic=geodesic)

    if scale is None:
        # scale = +infinity, the kernel is always equal to 1
        K_ij = constant_1_kernel(points=X, target_points=Y, **backend_args)

    else:
        if kernel == "gaussian":
            # Divisions are expensive: whenever possible, it's best to scale the points
            # ahead of time instead of scaling the distances for every pair of points.
            sqrt_2 = 1.41421356237
            X = X / (sqrt_2 * scale)
            Y = Y / (sqrt_2 * scale)

            # For dense computations, users may specify a cutoff as a kernel
            # value below which the kernel is assumed to be zero.
            # exp(-x^2) <= cutoff <=> x^2 >= -log(cutoff)
            if window is None and cutoff is not None and cutoff < 1:
                assert cutoff > 0
                backend_args["cutoff"] = -np.log(cutoff)

            K_ij = squared_distances(
                points=X,
                target_points=Y,
                kernel=lambda d2: (-d2).exp(),
                **backend_args,
            )

        elif kernel == "uniform":
            X = X / scale
            Y = Y / scale
            # For dense computations, users may specify a cutoff as a kernel
            # value below which the kernel is assumed to be zero.
            # For the uniform kernel, this just means discarding pairs of points
            # which are at distance > 1 (after rescaling).
            if window is None and cutoff is not None:
                backend_args["cutoff"] = 1.01  # To be on the safe side...

            K_ij = squared_distances(
                points=X,
                target_points=Y,
                kernel=lambda d2: 1.0 * (1 - d2).step(),
                **backend_args,
            )

    if normalize:

        total_weights_i = K_ij @ weights_j
        assert total_weights_i.shape == (M,)
        norm_i = 1.0 / total_weights_i

    else:
        norm_i = None

    assert K_ij.shape == (M, N)

    return LinearOperator(K_ij, input_scaling=weights_j, output_scaling=norm_i)


def convolution(
    *,
    source,
    target,
    kernel: Literal["uniform", "gaussian"] = "gaussian",
    scale: Optional[Number] = None,
    window: Literal[None, "ball", "knn", "spectral"] = None,
    cutoff: Optional[Number] = None,
    geodesic: bool = False,
    normalize: bool = False,
    dtype: Optional[Literal["float", "double"]] = None,
):
    """Same as PolyData.point_convolution, but with source and target as arguments."""

    N = source.n_points
    M = target.n_points

    weights_j = source.point_weights
    assert weights_j.shape == (N,)

    X = source.points
    Y = target.points

    if dtype == "float":
        X = X.float()
        Y = Y.float()
        weights_j = weights_j.float()
    elif dtype == "double":
        X = X.double()
        Y = Y.double()
        weights_j = weights_j.double()

    backend_args = dict(window=window, cutoff=cutoff, geodesic=geodesic)

    if scale is None:
        # scale = +infinity, the kernel is always equal to 1
        K_ij = constant_1_kernel(points=X, target_points=Y, **backend_args)

    else:
        if kernel == "gaussian":
            # Divisions are expensive: whenever possible, it's best to scale the points
            # ahead of time instead of scaling the distances for every pair of points.
            sqrt_2 = 1.41421356237
            X = X / (sqrt_2 * scale)
            Y = Y / (sqrt_2 * scale)

            # For dense computations, users may specify a cutoff as a kernel
            # value below which the kernel is assumed to be zero.
            # exp(-x^2) <= cutoff <=> x^2 >= -log(cutoff)
            if window is None and cutoff is not None and cutoff < 1:
                assert cutoff > 0
                backend_args["cutoff"] = -np.log(cutoff)

            K_ij = squared_distances(
                points=X,
                target_points=Y,
                kernel=lambda d2: (-d2).exp(),
                **backend_args,
            )

        elif kernel == "uniform":
            X = X / scale
            Y = Y / scale
            # For dense computations, users may specify a cutoff as a kernel
            # value below which the kernel is assumed to be zero.
            # For the uniform kernel, this just means discarding pairs of points
            # which are at distance > 1 (after rescaling).
            if window is None and cutoff is not None:
                backend_args["cutoff"] = 1.01  # To be on the safe side...

            K_ij = squared_distances(
                points=X,
                target_points=Y,
                kernel=lambda d2: 1.0 * (d2 <= 1),
                **backend_args,
            )

    if normalize:
        print(K_ij.shape)
        print(weights_j.shape)

        total_weights_i = K_ij @ weights_j
        assert total_weights_i.shape == (M,)
        norm_i = 1.0 / total_weights_i

    else:
        norm_i = None

    assert K_ij.shape == (M, N)

    return LinearOperator(K_ij, input_scaling=weights_j, output_scaling=norm_i)
