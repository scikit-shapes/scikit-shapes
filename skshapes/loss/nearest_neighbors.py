from pykeops.torch import Vi, Vj
import torch
from ..data import PolyData
from ..types import typecheck, Loss, FloatScalar


class NearestNeighborsLoss(Loss):
    @typecheck
    def __init__(self) -> None:
        pass

    @typecheck
    def __call__(self, source: PolyData, target: PolyData) -> FloatScalar:
        """
        Args:
            x (torch.Tensor): the current mesh
        Returns:
            loss (torch.Tensor): the data attachment loss
        """

        source_points = source.points
        target_points = target.points

        X_i = Vi(source_points)
        X_j = Vj(target_points)

        D_ij = ((X_i - X_j) ** 2).sum(-1)
        correspondences = D_ij.argKmin(K=1, dim=1).view(-1)

        return torch.norm(
            (source_points - target_points[correspondences]), p=2, dim=1
        ).mean()
