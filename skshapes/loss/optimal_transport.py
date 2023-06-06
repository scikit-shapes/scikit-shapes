from geomloss import SamplesLoss
import torch

from .._typing import *


class OptimalTransportLoss:
    @typecheck
    def __init__(self, loss: int = "sinkhorn", **kwargs) -> None:
        self.kwargs = kwargs
        self.loss = loss

    @typecheck
    def __call__(self, source: PolyDataType, target: PolyDataType) -> floatScalarType:

        source_triangles = source.triangles

        target_centers = target.triangle_centers
        target_weights = target.triangle_areas

        source_centers = source.triangle_centers
        source_weights = source.triangle_areas

        Loss = SamplesLoss(loss=self.loss, **self.kwargs)
        return Loss(source_weights, source_centers, target_weights, target_centers)


# class OptimalTransportLoss:
#     def __init__(self, *, loss="sinkhorn", **kwargs):
#         self.kwargs = kwargs
#         self.loss = loss

#     def fit(self, *, source, target):
#         self.source_triangles = source.triangles

#         self.target_centers = target.triangle_centers
#         self.target_weights = target.triangle_areas

#     def __call__(self, x):
#         """
#         Args:
#             x (torch.Tensor): the current mesh
#         Returns:
#             loss (torch.Tensor): the data attachment loss
#         """

#         A, B, C = (
#             x[self.source_triangles[0]],
#             x[self.source_triangles[1]],
#             x[self.source_triangles[2]],
#         )
#         centers = (A + B + C) / 3
#         weights = torch.cross(B - A, C - A).norm(dim=1) / 2

#         Loss = SamplesLoss(loss=self.loss, **self.kwargs)
#         return Loss(weights, centers, self.target_weights, self.target_centers)
