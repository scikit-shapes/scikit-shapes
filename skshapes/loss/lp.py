"""Lp losses for PolyData."""

import torch

from ..input_validation import typecheck
from ..types import FloatScalar, Number, polydata_type
from .baseloss import BaseLoss


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
        """Class constructor.

        Parameters
        ----------
        p
            the indice of the Lp Norm. Default to 2
        """
        if p <= 0:
            raise ValueError("p must be positive")

        self.p = p

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
        return torch.norm(source.points - target.points, p=self.p)


class L2Loss(BaseLoss):
    """L2 loss for PolyData.

    This class defines the L2 loss for PolyData. It is a wrapper around the
    LpLoss class with p=2.
    """

    def __new__(cls) -> LpLoss:
        """Create a new instance of the LpLoss class with p=2.

        Returns
        -------
        LpLoss
            the LpLoss class with p=2
        """
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

        Parameters
        ----------
        p
            the indice of the Lp Norm. Defaults to 2.
        """
        assert p > 0, "p must be positive"
        self.p = p

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
        return torch.norm(
            (source.landmark_points - target.landmark_points), p=self.p
        )
