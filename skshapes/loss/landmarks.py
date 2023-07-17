from typing import Any
import torch

from ..types import typecheck, Loss, PolyData, FloatScalar


class LandmarkLoss(Loss):
    @typecheck
    def __init__(self, p=2) -> None:
        self.p = p

    @typecheck
    def __call__(self, source: PolyData, target: PolyData) -> FloatScalar:
        return torch.norm(source.landmarks_3d - target.landmarks_3d, p=self.p)
