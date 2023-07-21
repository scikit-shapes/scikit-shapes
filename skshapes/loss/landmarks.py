from typing import Any
import torch
from ..data import PolyData

from ..types import typecheck, FloatScalar, Number
from .baseloss import BaseLoss


class LandmarkLoss(BaseLoss):
    @typecheck
    def __init__(self, p: Number = 2) -> None:
        self.p = p

    @typecheck
    def __call__(self, source: PolyData, target: PolyData) -> FloatScalar:
        return torch.norm(source.landmarks_3d - target.landmarks_3d, p=self.p)
