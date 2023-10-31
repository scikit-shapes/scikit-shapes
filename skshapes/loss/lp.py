"""Lp losses for PolyData."""

import torch
from ..types import typecheck, FloatScalar, Number, polydata_type
from .baseloss import BaseLoss


@typecheck
def _norm(x: torch.Tensor, p: Number) -> FloatScalar:
    """Compute the Lp norm of a tensor.

    Args:
        x (torch.Tensor): the tensor to compute the norm of
        p (Number): the indice of the Lp norm

    Returns:
        FloatScalar: the Lp norm of x
    """
    return torch.norm(x, p=p)


class LpLoss(BaseLoss):
    """Lp loss for PolyData.

    This class defines the L2 loss for PolyData. If X = (x_i) and Y = (y_i) are
    the points of two PolyData objects, the L2 loss is defined as:
    Lp(X, Y) = sum_i ||x_i - y_i||_p

    X and Y must have the same number of points. What is more, the points must
    be in correspondence, i.e. x_i and y_i must correspond to the same point.
    If this is not the case, the loss will be meaningless, consider using
    a loss function based on Optimal Transport or Nearest Neighbors instead.
    """

    @typecheck
    def __init__(self, p: Number = 2) -> None:
        """Constructor of the LpLoss class.

        Args:
            p (Number, optionnal): the indice of the Lp Norm. Default to 2
        """
        if p <= 0:
            raise ValueError("p must be positive")

        self.p = p

    @typecheck
    def __call__(
        self, source: polydata_type, target: polydata_type
    ) -> FloatScalar:
        """Compute the loss.

        Args:
            source (polydata_type): the source shape
            target (polydata_type): the target shape

        Returns:
            FloatScalar: the loss
        """
        super().__call__(source=source, target=target)
        return torch.norm(source.points - target.points, p=self.p)


class L2Loss(BaseLoss):
    """L2 loss for PolyData.

    This class defines the L2 loss for PolyData. It is a wrapper around the
    LpLoss class with p=2.
    """
    def __new__(cls) -> LpLoss:
        return LpLoss(p=2)


class LandmarkLoss(BaseLoss):
    """Landmark loss for PolyData.

    This class defines the Lp loss between the landmarks of two PolyData
    objects.
    If X = (x_i) and Y = (y_i) are the landmarks of two PolyData objects, the
    Lp loss is defined as:
    Lp(X, Y) = sum_i ||x_i - y_i||_p

    X and Y must have the same number of landmarks. What is more, the landmarks
    must be in correspondence, i.e. x_i and y_i must correspond to the same
    landmark. If this is not the case, the loss will be meaningless, consider
    using a loss function based on Optimal Transport or Nearest Neighbors
    instead.
    """

    @typecheck
    def __init__(self, p: Number = 2) -> None:
        """Initialize the LandmarkLoss class.

        Args:
            p (Number, optional): the indice of the Lp Norm. Defaults to 2.
        """
        assert p > 0, "p must be positive"
        self.p = p

    @typecheck
    def __call__(
        self, source: polydata_type, target: polydata_type
    ) -> FloatScalar:
        """Compute the loss.

        Args:
            source (polydata_type): the source shape
            target (polydata_type): the target shape

        Returns:
            FloatScalar: the loss
        """
        super().__call__(source=source, target=target)
        return _norm(
            x=(source.landmark_points - target.landmark_points), p=self.p
        )
