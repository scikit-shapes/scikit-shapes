"""Losses based on optimal transport for Polydata."""

from geomloss import SamplesLoss
from ..types import typecheck, FloatScalar, polydata_type
from typing import Literal
from .baseloss import BaseLoss

losses = Literal["sinkhorn", "hausdorff", "energy", "gaussian", "laplacian"]


class OptimalTransportLoss(BaseLoss):
    """Loss based on optimal transport for PolyData.

    This class defines a loss based on optimal transport for PolyData. More
    precisely, it initializes a SamplesLoss object from the geomloss library
    (https://www.kernel-operations.io/geomloss/). See the documentation of
    this library for more details. The default loss is the Sinkhorn loss.
    """
    @typecheck
    def __init__(self, loss: losses = "sinkhorn", **kwargs) -> None:
        """Initialize the OptimalTransportLoss class.

        For mor details on the arguments, see the documentation of geomloss:
        https://www.kernel-operations.io/geomloss/api/pytorch-api.html

        Args:
            loss (str, optional): The loss function to compute. Supported
                values are "sinkhorn", "hausdorff", "energy", "gaussian" and
                "laplacian". Defaults to "sinkhorn".
            **kwargs: additional arguments passed to the geomloss.SamplesLoss
        """
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

        Args:
            source (polydata_type): the source shape
            target (polydata_type): the target shape

        Returns:
            FloatScalar: the loss
        """
        super().__call__(source=source, target=target)
        target_points = target.points
        target_weights = target.point_weights

        source_points = source.points
        source_weights = source.point_weights

        Loss = SamplesLoss(loss=self.loss, **self.kwargs)
        return Loss(
            source_weights, source_points, target_weights, target_points
        )
