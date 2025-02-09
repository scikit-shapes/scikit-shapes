from pykeops.torch import LazyTensor

from ..input_validation import typecheck
from ..types import (
    Literal,
    Number,
    PointMasses,
    Points,
    PointVectorSignals,
)
from .neighborhoods import Neighborhoods


def keops_kernel(
    *,
    points: Points,
    window: Literal["gaussian kernel", "ball kernel"],
    scale: Number,
):

    # We scale the point clouds to have the simplest possible kernel formula
    if window == "gaussian kernel":
        sqrt_2 = 1.4142135623730951
        scaled_points = points / (sqrt_2 * scale)
    else:
        scaled_points = points / scale

    x_i = LazyTensor(scaled_points[:, None, :])  # (N, 1, D)
    x_j = LazyTensor(scaled_points[None, :, :])  # (1, N, D)
    # (N, N) symbolic matrix of squared distances
    D_ij = ((x_i - x_j) ** 2).sum(-1)

    if window == "gaussian kernel":
        K_ij = (-D_ij).exp()
    elif window == "ball kernel":
        K_ij = (1 - D_ij).step()

    return K_ij


class KernelNeighborhoods(Neighborhoods):

    @typecheck
    def __init__(
        self,
        *,
        points: Points,
        masses: PointMasses,
        window: Literal["gaussian kernel", "ball kernel"],
        scale: Number,
        n_normalization_iterations: int | None = None,
        smoothing_method: Literal[
            "auto", "exact", "exp(x)=1/(1-x)", "nystroem"
        ] = "auto",
        laplacian_method: Literal["auto", "exact", "log(x)=x-1"] = "auto",
    ):
        super().__init__(
            masses=masses,
            scale=scale,
            n_normalization_iterations=n_normalization_iterations,
            smoothing_method=smoothing_method,
            laplacian_method=laplacian_method,
        )
        if scale <= 0:
            msg = f"scale must be > 0, received {scale}."
            raise ValueError(msg)

        self.K_ij = keops_kernel(
            points=points,
            window=window,
            scale=scale,
        )
        assert self.K_ij.shape == (self.n_points, self.n_points)

        self._compute_scaling()

    @typecheck
    def _smooth_without_scaling(
        self, signal: PointVectorSignals
    ) -> PointVectorSignals:
        assert signal.shape[0] == self.n_points
        return self.K_ij @ signal
