"""Lp losses for PolyData."""

import torch

from ..input_validation import typecheck
from ..types import FloatScalar, Number, polydata_type
from .baseloss import BaseLoss


class LpLoss(BaseLoss):
    """Lp loss for PolyData.

    This class defines the L2 loss for PolyData. If X = (x_i) and Y = (y_i) are
    the points of two PolyData objects, the Lp loss is defined as:
    Lp(X, Y) = sum_i ||x_i - y_i|| ^ p where ||.|| is the Euclidean norm.

    X and Y must have the same number of points. What is more, the points must
    be in correspondence, i.e. x_i and y_i must correspond to the same point.
    If this is not the case, the loss will be meaningless, consider using
    a loss function based on Optimal Transport or Nearest Neighbors instead.

    Parameters
    ----------
    p
        the indice of the Lp Norm. Default to 2.
    """

    @typecheck
    def __init__(self, p: Number = 2) -> None:
        if p < 0:
            msg = "p must be nonnegative"
            raise ValueError(msg)

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
        sqdists = ((source.points - target.points) ** 2).sum(dim=-1)
        if self.p == 2:
            distsp = sqdists
        elif self.p == 1:
            distsp = sqdists.sqrt()
        else:
            distsp = sqdists ** (self.p / 2)
        return distsp.sum()


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
    If $X = (x_i)$ and $Y = (y_i)$ are the landmarks of two PolyData objects, the
    Lp loss is defined as:

    $$Lp(X, Y) = \sum_i \Vert x_i - y_i \Vert_p$$

    X and Y must have the same number of landmarks. What is more, the landmarks
    must be in correspondence, i.e. $x_i$ and $y_i$ must correspond to the same
    landmark. If this is not the case, the loss will be meaningless, consider
    using a loss function based on Optimal Transport or Nearest Neighbors
    instead.

    Parameters
    ----------
    p
        the indice of the Lp Norm. Defaults to 2.
    """

    @typecheck
    def __init__(self, p: Number = 2) -> None:
        """Initialize the LandmarkLoss class."""
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
