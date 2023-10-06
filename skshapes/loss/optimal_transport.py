"""Optimal transport loss for PolyData."""

from geomloss import SamplesLoss
import torch

from ..types import typecheck, FloatScalar, polydata_type
from .baseloss import BaseLoss


class OptimalTransportLoss(BaseLoss):
    @typecheck
    def __init__(self, loss: str = "sinkhorn", **kwargs) -> None:
        self.kwargs = kwargs
        self.loss = loss

    @typecheck
    def __call__(self, source: polydata_type, target: polydata_type) -> FloatScalar:
        target_centers = target.triangle_centers
        target_weights = target.triangle_areas

        source_centers = source.triangle_centers
        source_weights = source.triangle_areas

        Loss = SamplesLoss(loss=self.loss, **self.kwargs)
        return Loss(source_weights, source_centers, target_weights, target_centers)
