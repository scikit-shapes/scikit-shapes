"""Nearest Neighbors loss for PolyData."""

import torch
from pykeops.torch import Vi, Vj

from ..input_validation import typecheck
from ..types import FloatScalar, polydata_type
from .baseloss import BaseLoss


class NearestNeighborsLoss(BaseLoss):
    """Loss based on nearest neighbors for PolyData.

    This class defines a loss corresponding to the nearest neighbors distance
    between the points of two PolyData objects. More precisely, for each point
    in the source PolyData, we compute the distance to its nearest neighbor in
    the target PolyData. The loss is then the average of these distances.

    The distances are computed using the lazy tensor library pykeops :
    https://www.kernel-operations.io/keops/index.html
    """

    @typecheck
    def __init__(self) -> None:
        pass

    @typecheck
    def __call__(
        self, source: polydata_type, target: polydata_type
    ) -> FloatScalar:
        """Compute the loss.

        Parameters
        ----------
        source
            the source shape
        target
            the target shape

        Returns
        -------
        FloatScalar
            the loss
        """
        super().__call__(source=source, target=target)

        source_points = source.points
        target_points = target.points

        X_i = Vi(source_points)
        X_j = Vj(target_points)

        D_ij = ((X_i - X_j) ** 2).sum(-1)
        correspondences = D_ij.argKmin(K=1, dim=1).view(-1)

        return torch.norm(
            (source_points - target_points[correspondences]), p=2, dim=1
        ).mean()
