"""Point convolutions kernel."""

from typing import Literal

import numpy as np

from ..input_validation import typecheck
from ..types import Number, polydata_type
from .constant_kernel import constant_1_kernel
from .linear_operator import LinearOperator
from .squared_distances import squared_distances


@typecheck
def _point_convolution(
    self,
    *,
    kernel: Literal["uniform", "gaussian"] = "gaussian",
    scale: Number | None = None,
    window: Literal["ball", "knn", "spectral"] | None = None,
    cutoff: Number | None = None,
    geodesic: bool = False,
    normalize: bool = False,
    dtype: Literal["float", "double"] | None = None,
    target: polydata_type | None = None,
) -> LinearOperator:
    """Convolution kernel on a PolyData.

    Creates a convolution kernel on a PolyData as a (N, N) linear operatorif no
    target is provided, or as a (M, N) linear operator if a target is provided.

    Parameters
    ----------
    kernel
        The kernel to use.
    scale
        The scale of the kernel.
    window
        The type of window to use.
    cutoff
        The cutoff value for the window.
    geodesic
        Whether to use geodesic distances.
    normalize
        Whether to normalize the kernel.
    dtype
        The data type of the kernel.
    target
        The target PolyData.

    Returns
    -------
    LinearOperator
        A (N, N) or (M, N) convolution kernel.

    """
    if target is None:
        target = self

    N = self.n_points
    M = target.n_points

    weights_j = self.point_masses
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

    backend_args = {"window": window, "cutoff": cutoff, "geodesic": geodesic}

    if scale is None:
        # scale = +infinity, the kernel is always equal to 1
        K_ij = constant_1_kernel(points=X, target_points=Y, **backend_args)

    elif kernel == "gaussian":
        # Divisions are expensive: whenever possible, it's best to scale
        # the points ahead of time instead of scaling the distances for
        # every pair of points.
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
        # For the uniform kernel, this just means discarding pairs of
        # points which are at distance > 1 (after rescaling).
        if window is None and cutoff is not None:
            backend_args["cutoff"] = 1.01  # To be on the safe side...

        K_ij = squared_distances(
            points=X,
            target_points=Y,
            kernel=lambda d2: 1.0 * (1 - d2).step(),
            **backend_args,
        )

    if normalize:
        # Clip to avoid division by zero
        total_weights_i = (K_ij @ weights_j).clip(min=1e-6)
        assert total_weights_i.shape == (M,)
        norm_i = 1.0 / total_weights_i

    else:
        norm_i = None

    assert K_ij.shape == (M, N)

    return LinearOperator(K_ij, input_scaling=weights_j, output_scaling=norm_i)
