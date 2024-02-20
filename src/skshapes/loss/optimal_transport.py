"""Losses based on optimal transport for Polydata."""

from typing import Literal

from geomloss import SamplesLoss

from ..input_validation import typecheck
from ..types import FloatScalar, polydata_type
from .baseloss import BaseLoss

losses = Literal["sinkhorn", "hausdorff", "energy", "gaussian", "laplacian"]


class OptimalTransportLoss(BaseLoss):
    """Loss based on optimal transport for PolyData.

    This class defines a loss based on optimal transport for PolyData. More
    precisely, it initializes a SamplesLoss object from the geomloss library.
    See its [documentation]((https://www.kernel-operations.io/geomloss/))
    for more details. The default loss is the Sinkhorn loss.


    Parameters
    ----------
    loss
        The loss function to compute. Supported values are "sinkhorn",
        "hausdorff", "energy", "gaussian" and "laplacian".
    **kwargs
        additional arguments passed to the geomloss.SamplesLoss object.

    """

    @typecheck
    def __init__(self, loss: losses = "sinkhorn", **kwargs) -> None:
        self.kwargs = kwargs
        self.loss = loss

    @typecheck
    def __call__(
        self, source: polydata_type, target: polydata_type
    ) -> FloatScalar:
        """Compute the loss.

        Depending on the topology of the source and the target shapes,
        the loss is computed in different ways. Source and target are first
        converted to WeightedPoints objects, and then the loss is computed
        using the geomloss library.

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
        target_points = target.points
        target_weights = target.point_weights
        target_weights = target_weights / target_weights.sum()

        source_points = source.points
        source_weights = source.point_weights
        source_weights = source_weights / source_weights.sum()

        Loss = SamplesLoss(loss=self.loss, **self.kwargs)
        return Loss(
            source_weights, source_points, target_weights, target_points
        )
