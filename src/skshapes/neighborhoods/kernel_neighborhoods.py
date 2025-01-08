from pykeops.torch import LazyTensor

from ..input_validation import typecheck
from ..types import Literal, PointAnySignals, Points, PointWeights
from .neighborhoods import Neighborhoods


def keops_kernel(
    points: Points,
    window: Literal["gaussian", "ball"],
    radius: float,
):

    # We scale the point clouds to have the simplest possible kernel formula
    if window == "gaussian":
        sqrt_2 = 1.4142135623730951
        scaled_points = points / (sqrt_2 * radius)
    else:
        scaled_points = points / radius

    x_i = LazyTensor(scaled_points[:, None, :])  # (N, 1, D)
    x_j = LazyTensor(scaled_points[None, :, :])  # (1, N, D)
    D_ij = ((x_i - x_j) ** 2).sum(
        -1
    )  # (N, N) symbolic matrix of squared distances

    if window == "gaussian":
        K_ij = (-D_ij).exp()
    elif window == "ball":
        K_ij = (1 - D_ij).step()

    return K_ij


class KernelNeighborhoods(Neighborhoods):

    @typecheck
    def __init__(
        self,
        *,
        points: Points,
        point_weights: PointWeights | None = None,
        window: Literal["gaussian", "ball"],
        radius: float,
    ):
        super().__init__(point_weights=point_weights)
        if radius <= 0:
            msg = f"radius must be > 0, received {radius}."
            raise ValueError(msg)

        self.n_points = points.shape[0]
        self.K_ij = keops_kernel(points, window, radius)
        assert self.K_ij.shape == (self.n_points, self.n_points)

    @typecheck
    def _convolve_without_weights(
        self,
        *,
        signal: PointAnySignals,
    ):
        assert signal.shape[0] == self.n_points
        flat_signal = signal.view(self.n_points, -1)
        flat_output = self.K_ij @ flat_signal
        return flat_output.view_as(signal)
